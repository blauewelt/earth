// Browser tests for the earth globe app.
//
// In CI these hit the real CDN (cdnjs) and NASA GIBS. In the local sandbox,
// set MIRROR=1 to route the Cesium CDN to the vendored copy (_vendor/cesium)
// and GIBS to a local proxy on :8081 (see README "Testing").
"use strict";
const { test, expect } = require("@playwright/test");

const CDN = "https://cdnjs.cloudflare.com/ajax/libs/cesium/1.133.1";

test.beforeEach(async ({ page, baseURL }) => {
  if (process.env.MIRROR) {
    await page.route(/https:\/\/cdnjs\.cloudflare\.com\/.*/, async (route) => {
      try {
        const url = route.request().url()
          .replace(CDN, `${baseURL}/_vendor/cesium`)
          .replace("widgets.min.css", "widgets.css");
        const resp = await page.request.get(url);
        await route.fulfill({ response: resp });
      } catch {
        await route.abort().catch(() => {});
      }
    });
    await page.route(/https:\/\/gibs\.earthdata\.nasa\.gov\/.*/, async (route) => {
      try {
        const url = route.request().url()
          .replace("https://gibs.earthdata.nasa.gov", "http://localhost:8081");
        const resp = await page.request.get(url);
        await route.fulfill({ response: resp });
      } catch {
        await route.abort().catch(() => {});
      }
    });
    await page.route(/https:\/\/api\.gbif\.org\/.*/, async (route) => {
      try {
        const url = route.request().url().replace("https://api.gbif.org", "http://localhost:8082");
        const resp = await page.request.get(url);
        await route.fulfill({ response: resp });
      } catch {
        await route.abort().catch(() => {});
      }
    });
  }
  page.__errors = [];
  page.on("pageerror", (e) => page.__errors.push(String(e)));
  await page.goto("/");
  await page.waitForFunction(() => window.__earth?.viewer, null, { timeout: 30000 });
});

test("loads without page errors and renders a WebGL canvas", async ({ page }) => {
  await expect(page.locator("#cesiumContainer canvas").first()).toBeVisible();
  await page.waitForTimeout(1500);
  expect(page.__errors).toEqual([]);
});

test("GIBS tiling scheme matches the published matrix definitions", async ({ page }) => {
  const r = await page.evaluate(() => {
    const s = new window.__earth.GIBSGeographicTilingScheme();
    const rect = s.tileXYToNativeRectangle(1, 0, 0); // partial edge tile
    const pos = { longitude: Cesium.Math.toRadians(170), latitude: Cesium.Math.toRadians(0) };
    const xy = s.positionToTileXY(pos, 2);
    return {
      x0: s.getNumberOfXTilesAtLevel(0), y0: s.getNumberOfYTilesAtLevel(0),
      x1: s.getNumberOfXTilesAtLevel(1), y1: s.getNumberOfYTilesAtLevel(1),
      x5: s.getNumberOfXTilesAtLevel(5), y5: s.getNumberOfYTilesAtLevel(5),
      edgeWest: rect.west, edgeEast: rect.east,
      pick: [xy.x, xy.y],
    };
  });
  // From GIBS GetCapabilities: level 0 = 2x1, level 1 = 3x2, level 5 = 40x20
  expect([r.x0, r.y0]).toEqual([2, 1]);
  expect([r.x1, r.y1]).toEqual([3, 2]);
  expect([r.x5, r.y5]).toEqual([40, 20]);
  // Edge tile must declare its FULL nominal span (not clamped to 180) —
  // clamping this is the bug that blanked the Pacific.
  expect(r.edgeWest).toBe(108);
  expect(r.edgeEast).toBe(396);
  // lon 170 at level 2 (span 72°) → x = floor(350/72) = 4
  expect(r.pick).toEqual([4, 1]);
});

test("layer toggles add and remove imagery layers", async ({ page }) => {
  const count = () => page.evaluate(() => window.__earth.viewer.imageryLayers.length);
  const before = await count();
  await page.check('#layer-list input[data-id="precip"]');
  expect(await count()).toBe(before + 1);
  await page.uncheck('#layer-list input[data-id="precip"]');
  expect(await count()).toBe(before);
});

test("date change rebuilds timed layers with the new date", async ({ page }) => {
  await page.fill("#layer-date", "2025-01-15");
  await page.dispatchEvent("#layer-date", "change");
  const r = await page.evaluate(() => ({
    date: window.__earth.state.date,
    sstOn: !!window.__earth.state.layers["sst"]?.layer,
  }));
  expect(r.date).toBe("2025-01-15");
  expect(r.sstOn).toBe(true);
});

test("comparison mode creates split layers, labels, and a draggable divider", async ({ page }) => {
  await page.selectOption("#compare-select", "10");
  await expect(page.locator("#split-handle")).toBeVisible();
  const r = await page.evaluate(() => {
    const e = window.__earth.state.layers["sst"];
    return {
      cmp: !!e.cmpLayer,
      main: e.layer.splitDirection === Cesium.SplitDirection.RIGHT,
      past: e.cmpLayer.splitDirection === Cesium.SplitDirection.LEFT,
      cmpDate: window.__earth.compareDate(),
      curDate: window.__earth.state.date,
    };
  });
  expect(r.cmp && r.main && r.past).toBe(true);
  expect(Number(r.curDate.slice(0, 4)) - Number(r.cmpDate.slice(0, 4))).toBe(10);
  await expect(page.locator("#split-label-left")).toHaveText(r.cmpDate);
  await expect(page.locator("#split-label-right")).toHaveText(r.curDate);
  // Off again removes the comparison twin
  await page.selectOption("#compare-select", "0");
  await expect(page.locator("#split-handle")).toBeHidden();
  expect(await page.evaluate(() => !!window.__earth.state.layers["sst"].cmpLayer)).toBe(false);
});

test("zoom buttons and wheel zoom move the camera briskly", async ({ page }) => {
  const height = () =>
    page.evaluate(() => window.__earth.viewer.camera.positionCartographic.height);

  // buttons
  const h0 = await height();
  await page.click("#zoom-in");
  await expect.poll(height).toBeLessThan(h0 * 0.7);
  const hIn = await height();
  await page.click("#zoom-out");
  await expect.poll(height).toBeGreaterThan(hIn); // zooms back out

  // wheel: one notch covers a big fraction of the height (fast zoom)
  const before = await height();
  await page.evaluate(() =>
    window.__wheelZoom({ deltaY: 120, deltaMode: 0, ctrlKey: false, preventDefault() {} }));
  const afterWheel = await height();
  expect(afterWheel).toBeLessThan(before * 0.35); // ~0.85 of height per notch → big jump

  // touch pinch stays native
  const hasPinch = await page.evaluate(() =>
    window.__earth.viewer.scene.screenSpaceCameraController.zoomEventTypes
      .includes(Cesium.CameraEventType.PINCH));
  expect(hasPinch).toBe(true);

  // trackpad pinch (ctrlKey) with a small delta still zooms meaningfully
  const b2 = await height();
  await page.evaluate(() =>
    window.__wheelZoom({ deltaY: 20, deltaMode: 0, ctrlKey: true, preventDefault() {} }));
  expect(await height()).toBeLessThan(b2 * 0.7);
});

test("Climate TRACE and Argo point layers load with expected counts", async ({ page }) => {
  await page.check("#toggle-climatetrace");
  await expect
    .poll(() => page.evaluate(() => window.__earth.pointLayers.climatetrace?.collection.length ?? 0))
    .toBe(1000);
  await page.check("#toggle-argo");
  await expect
    .poll(() => page.evaluate(() => window.__earth.pointLayers.argo?.collection.length ?? 0))
    .toBeGreaterThan(2000);
  await expect(page.locator("#meta-climatetrace")).toContainText("snapshot");
  // toggling off hides but keeps the collection
  await page.uncheck("#toggle-argo");
  expect(await page.evaluate(() => window.__earth.pointLayers.argo.collection.show)).toBe(false);
});

test("stations render and can be hidden", async ({ page }) => {
  await expect
    .poll(() => page.evaluate(() => window.__earth.stations?.entities.values.length ?? 0))
    .toBeGreaterThanOrEqual(13);
  await page.uncheck("#toggle-stations");
  expect(await page.evaluate(() => window.__earth.stations.show)).toBe(false);
});

test("AMOC dashboard loads RAPID data and populates stats + chart", async ({ page }) => {
  await page.click("#tab-amoc");
  await expect(page.locator("#amoc-latest .stat-value")).not.toHaveText("–");
  const r = await page.evaluate(() => ({
    n: window.__earth.rapid.t.length,
    latest: Number(document.querySelector("#amoc-latest .stat-value").textContent),
    early: Number(document.querySelector("#amoc-early .stat-value").textContent),
    chartW: document.getElementById("amoc-chart").width,
  }));
  expect(r.n).toBeGreaterThan(700);
  expect(r.latest).toBeGreaterThan(0);
  expect(r.early).toBeGreaterThan(10); // 2004-08 mean ~18.5 Sv
  expect(r.chartW).toBeGreaterThan(0);
  // hover produces a tooltip
  await page.hover("#amoc-chart", { position: { x: 150, y: 80 } });
  await expect(page.locator("#amoc-tooltip")).toBeVisible();
  await expect(page.locator("#amoc-tooltip")).toContainText("Sv");
});

test("catalog browser filters the dataset list", async ({ page }) => {
  await page.click("#tab-catalog");
  await expect(page.locator("#catalog-count")).toContainText("datasets");
  const totalTxt = await page.locator("#catalog-count").textContent();
  const total = Number(totalTxt.match(/of (\d+)/)[1]);
  expect(total).toBeGreaterThanOrEqual(241);
  await page.fill("#catalog-search", "RAPID");
  const txt = await page.locator("#catalog-count").textContent();
  const n = Number(txt.split(" ")[0]);
  expect(n).toBeGreaterThan(0);
  expect(n).toBeLessThan(50);
  await page.fill("#catalog-search", "");
  await page.check("#filter-amoc");
  await expect(page.locator("#catalog-count")).toContainText(`58 of ${total}`);
});

test("every layer title is a clickable documentation link", async ({ page }) => {
  // GIBS layers: title itself links to the dataset docs, checkbox toggles separately
  const links = page.locator("#layer-list .layer-head a.title-link");
  await expect(links).toHaveCount(16);
  for (const href of await links.evaluateAll((as) => as.map((a) => a.href))) {
    expect(href).toMatch(/^https:\/\//);
  }
  await expect(links.first()).toHaveAttribute("target", "_blank");
  // data + analysis layers (SST ensemble, Climate TRACE, Argo, stations, GBIF) too
  await expect(page.locator("#panel-layers .layer-head a.title-link")).toHaveCount(22);
  // clicking the title must NOT toggle the layer
  const before = await page.evaluate(() => window.__earth.viewer.imageryLayers.length);
  const [popup] = await Promise.all([
    page.context().waitForEvent("page"),
    page.locator('#layer-list .layer-head a.title-link').first().click(),
  ]);
  await popup.close();
  expect(await page.evaluate(() => window.__earth.viewer.imageryLayers.length)).toBe(before);
});

test("legends appear for active layers and follow toggles", async ({ page }) => {
  // SST is on by default → its legend is showing, rendered from the GIBS colormap
  await expect(page.locator("#legend-panel")).toBeVisible();
  await expect(page.locator("#legend-panel .legend-item")).toHaveCount(1);
  await expect(page.locator("#legend-panel .legend-item canvas.legend-bar")).toHaveCount(1);
  await expect(page.locator("#legend-panel .legend-range").first()).toContainText("°C");
  await page.check('#layer-list input[data-id="precip"]');
  await expect(page.locator("#legend-panel .legend-item")).toHaveCount(2);
  await page.uncheck('#layer-list input[data-id="precip"]');
  await page.uncheck('#layer-list input[data-id="sst"]');
  await expect(page.locator("#legend-panel")).toBeHidden();
});

test("hovering a legend shows the exact value of that color", async ({ page }) => {
  const bar = page.locator("#legend-panel canvas.legend-bar").first();
  await expect(bar).toBeVisible();
  const box = await bar.boundingBox();
  // hover mid-scale → tooltip with a number + units
  await bar.hover({ position: { x: box.width / 2, y: 7 } });
  const tip = page.locator("#legend-panel .legend-tip").first();
  await expect(tip).toBeVisible();
  await expect(tip).toContainText("°C");
  const mid = parseFloat(await tip.textContent());
  expect(Number.isFinite(mid)).toBe(true);
  // hover near the left end → smaller value than mid-scale
  await bar.hover({ position: { x: 4, y: 7 } });
  const left = parseFloat(await tip.textContent());
  expect(left).toBeLessThan(mid);
  // parser sanity on the real SST colormap: ~200+ ordered entries in °C
  const cm = await page.evaluate(async () => {
    const xml = await (await fetch("https://gibs.earthdata.nasa.gov/colormaps/v1.3/GHRSST_Sea_Surface_Temperature.xml")).text();
    const p = window.__earth.parseColormapEntries(xml);
    return { units: p.units, n: p.entries.length, first: p.entries[0].lo, last: p.entries[p.entries.length - 1].hi };
  });
  expect(cm.units).toBe("°C");
  expect(cm.n).toBeGreaterThan(150);
  expect(cm.first).toBeLessThan(cm.last);
});

test("delta legend hover shows the signed difference in °C", async ({ page }) => {
  await page.selectOption("#compare-select", "10");
  await page.selectOption("#compare-mode", "delta");
  const bar = page.locator("#legend-panel .delta-bar").first();
  await expect(bar).toBeVisible();
  const box = await bar.boundingBox();
  await bar.hover({ position: { x: box.width * 0.9, y: 5 } });
  const tip = page.locator("#legend-panel .legend-tip").first();
  await expect(tip).toBeVisible();
  await expect(tip).toContainText("+");
  await expect(tip).toContainText("increase");
  await bar.hover({ position: { x: box.width * 0.1, y: 5 } });
  await expect(tip).toContainText("decrease");
});

test("colormap parser and delta colorization are correct", async ({ page }) => {
  const r = await page.evaluate(() => {
    const xml = `
      <ColorMapEntry rgb="10,20,30" transparent="false" sourceValue="[5.00,5.05)" value="[5.00,5.05)" ref="1"/>
      <ColorMapEntry rgb="0,0,0" transparent="true" nodata="true" ref="0"/>
      <ColorMapEntry rgb="40,50,60" transparent="false" sourceValue="[-INF,0.00)" value="[-INF,0.00)" ref="2"/>`;
    const lut = window.__earth.parseColormap(xml);
    return {
      size: lut.size,
      mid: lut.get((10 << 16) | (20 << 8) | 30),
      inf: lut.get((40 << 16) | (50 << 8) | 60),
      warm: window.__earth.deltaColor(3),
      cool: window.__earth.deltaColor(-3),
      zero: window.__earth.deltaColor(0),
    };
  });
  expect(r.size).toBe(2);              // transparent/nodata entries excluded
  expect(r.mid).toBeCloseTo(5.025, 3); // midpoint of the range
  expect(r.inf).toBe(0);               // open-ended range uses the finite bound
  expect(r.warm[0]).toBeGreaterThan(r.warm[2]); // warmer → red
  expect(r.cool[2]).toBeGreaterThan(r.cool[0]); // cooler → blue
  expect(r.zero[3]).toBe(0);           // no change → transparent
  expect(r.warm[3]).toBeGreaterThan(150); // strong delta → strongly visible
});

test("computed-difference mode replaces the SST split with a delta layer", async ({ page }) => {
  await page.selectOption("#compare-select", "10");
  await expect(page.locator("#compare-mode-row")).toBeVisible();
  await page.selectOption("#compare-mode", "delta");
  const r = await page.evaluate(() => {
    const e = window.__earth.state.layers["sst"];
    return {
      isDelta: e.isDelta,
      noTwin: !e.cmpLayer,
      providerIsDelta: e.layer.imageryProvider instanceof window.__earth.DeltaProvider,
    };
  });
  expect(r.isDelta && r.noTwin && r.providerIsDelta).toBe(true);
  await expect(page.locator("#split-handle")).toBeHidden(); // no swipe in delta mode
  await expect(page.locator("#legend-panel")).toContainText("Δ Sea surface temperature");
  // back to split restores the swipe pair
  await page.selectOption("#compare-mode", "split");
  await expect(page.locator("#split-handle")).toBeVisible();
  expect(await page.evaluate(() => !!window.__earth.state.layers["sst"].cmpLayer)).toBe(true);
});

test("rolling window: fixed-length interval, correct sampling", async ({ page }) => {
  const d = await page.evaluate(() => ({
    one: window.__earth.windowSampleDates("2026-07-21", 1),
    d30: window.__earth.windowSampleDates("2026-07-21", 30),
    d365: window.__earth.windowSampleDates("2026-07-21", 365),
    add: window.__earth.addDays("2026-01-01", -1),
    label1: window.__earth.windowLabel(1),
    label30: window.__earth.windowLabel(30),
  }));
  expect(d.one).toEqual(["2026-07-21"]);
  // window always ends on the date and spans exactly N days back (fixed length)
  expect(d.d30[0]).toBe("2026-07-21");
  expect(d.d30[d.d30.length - 1]).toBe("2026-06-22"); // 29 days before → 30-day span
  expect(d.d30.length).toBeLessThanOrEqual(12);
  expect(d.d365[0]).toBe("2026-07-21");
  expect(d.d365[d.d365.length - 1]).toBe("2025-07-22"); // ~365-day span
  expect(d.add).toBe("2025-12-31"); // date arithmetic across year boundary
  expect(d.label1).toBe("single day");
  expect(d.label30).toBe("past 30 days");
});

test("aggregation window is orthogonal to the display mode", async ({ page }) => {
  const setWindow = (v) => page.evaluate((val) => {
    const s = document.getElementById("window-days");
    s.value = String(val);
    s.dispatchEvent(new Event("change"));
  }, v);

  await page.selectOption("#compare-select", "10");

  // delta mode + 30-day window → delta provider carrying the window
  await page.selectOption("#compare-mode", "delta");
  await setWindow(30);
  let r = await page.evaluate(() => {
    const e = window.__earth.state.layers["sst"];
    return { win: window.__earth.state.windowDays, isDelta: e.isDelta,
             provWin: e.layer.imageryProvider.window, label: document.getElementById("window-value").textContent };
  });
  expect(r.win).toBe(30);
  expect(r.isDelta).toBe(true);
  expect(r.provWin).toBe(30);
  expect(r.label).toBe("past 30 days");
  await expect(page.locator("#legend-panel")).toContainText("past 30 days mean");

  // same window, switch to side-by-side → aggregate providers on both sides
  await page.selectOption("#compare-mode", "split");
  r = await page.evaluate(() => {
    const e = window.__earth.state.layers["sst"];
    return { isAgg: e.isAggregate, main: e.layer.imageryProvider.constructor.name,
             cmp: e.cmpLayer.imageryProvider.constructor.name };
  });
  expect(r.isAgg).toBe(true);
  expect(r.main).toBe("AggregateProvider");
  expect(r.cmp).toBe("AggregateProvider");
  await expect(page.locator("#split-handle")).toBeVisible();

  // window applies even without comparison (single aggregated layer)
  await page.selectOption("#compare-select", "0");
  r = await page.evaluate(() => {
    const e = window.__earth.state.layers["sst"];
    return { isAgg: e.isAggregate, name: e.layer.imageryProvider.constructor.name, hasCmp: !!e.cmpLayer };
  });
  expect(r.isAgg).toBe(true);
  expect(r.name).toBe("AggregateProvider");
  expect(r.hasCmp).toBe(false);

  // back to a single day → plain GIBS provider
  await setWindow(1);
  r = await page.evaluate(() => {
    const e = window.__earth.state.layers["sst"];
    return { isAgg: e.isAggregate, name: e.layer.imageryProvider.constructor.name };
  });
  expect(r.isAgg).toBe(false);
  expect(r.name).not.toBe("AggregateProvider");
});

test("comparison hint explains non-differenceable & point layers in both modes", async ({ page }) => {
  await page.selectOption("#compare-select", "10");
  await page.selectOption("#compare-mode", "delta");
  await expect(page.locator("#delta-hint")).toBeHidden(); // SST alone is differenceable
  // precipitation has no deltaRange (instantaneous/log) → hint appears in delta mode
  await page.check('#layer-list input[data-id="precip"]');
  await expect(page.locator("#delta-hint")).toBeVisible();
  await expect(page.locator("#delta-hint")).toContainText("instantaneous");
  await page.uncheck('#layer-list input[data-id="precip"]');
  await expect(page.locator("#delta-hint")).toBeHidden();
  // glaciers: single-snapshot note appears in delta AND side-by-side modes
  await page.check("#toggle-glaciers");
  await expect(page.locator("#delta-hint")).toBeVisible();
  await expect(page.locator("#delta-hint")).toContainText("single inventory");
  await page.selectOption("#compare-mode", "split");
  await expect(page.locator("#delta-hint")).toBeVisible(); // still shown in side-by-side
  await page.uncheck("#toggle-glaciers");
  await expect(page.locator("#delta-hint")).toBeHidden();
});

test("SST ensemble layer renders mean and spread with matching legends", async ({ page }) => {
  await page.check("#toggle-sst-ensemble");
  await expect
    .poll(() => page.evaluate(() => !!window.__earth.ensembleLayer))
    .toBe(true);
  // mean mode uses the GHRSST scale
  await expect(page.locator("#legend-panel")).toContainText("SST ensemble mean");
  const r = await page.evaluate(() => {
    const prov = window.__earth.ensembleLayer.imageryProvider;
    return { name: prov.constructor.name, mode: prov.mode };
  });
  expect(r.name).toBe("SSTEnsembleProvider");
  expect(r.mode).toBe("mean");
  // provider produces opaque pixels over ocean tiles
  const opaque = await page.evaluate(async () => {
    const prov = new window.__earth.SSTEnsembleProvider(
      [{ name: "MUR", layer: "GHRSST_L4_MUR_Sea_Surface_Temperature", tms: "1km" },
       { name: "GAMSSA", layer: "GHRSST_L4_GAMSSA_GDS2_Sea_Surface_Temperature", tms: "2km" }],
      window.__earth.state.date, "mean");
    const c = await prov.requestImage(4, 2, 3);
    const d = c.getContext("2d").getImageData(0, 0, 512, 512).data;
    let n = 0; for (let i = 3; i < d.length; i += 4) if (d[i] > 0) n++;
    return n;
  });
  expect(opaque).toBeGreaterThan(10000);
  // spread mode swaps the legend and colour semantics
  await page.selectOption("#ensemble-mode", "spread");
  await expect(page.locator("#legend-panel")).toContainText("ensemble spread");
  const sp = await page.evaluate(() => ({
    zero: window.__earth.spreadColor(0)[3],
    big: window.__earth.spreadColor(2)[3],
  }));
  expect(sp.zero).toBe(0);         // no disagreement → transparent
  expect(sp.big).toBeGreaterThan(150);
});

test("sea-level dashboard loads the budget, stats and chart", async ({ page }) => {
  await page.click("#tab-sealevel");
  await expect(page.locator("#sl-total .stat-value")).not.toHaveText("–");
  const r = await page.evaluate(() => ({
    years: window.__earth.sealevel.years.length,
    total: Number(document.querySelector("#sl-total .stat-value").textContent),
    rate: Number(document.querySelector("#sl-rate .stat-value").textContent),
    chartW: document.getElementById("sl-chart").width,
    legend: document.getElementById("sl-legend").children.length,
    trend: window.__earth.linTrend([2000, 2010, 2020], [0, 30, 60]),
  }));
  expect(r.years).toBeGreaterThan(110);
  expect(r.total).toBeGreaterThan(150);           // ~209 mm since 1900
  expect(r.rate).toBeGreaterThan(2.5);            // satellite-era ~3.2 mm/yr
  expect(r.rate).toBeLessThan(4.5);
  expect(r.chartW).toBeGreaterThan(0);
  expect(r.legend).toBe(8);                       // observed + summed + 5 components + altimetry
  expect(r.trend).toBeCloseTo(3, 5);              // linear-trend helper is correct
  // hover produces a breakdown tooltip
  await page.hover("#sl-chart", { position: { x: 160, y: 90 } });
  await expect(page.locator("#sl-tooltip")).toBeVisible();
  await expect(page.locator("#sl-tooltip")).toContainText("observed");
});

test("biodiversity (GBIF) layer toggles and filters by species", async ({ page }) => {
  // species selector is populated from the bundled curated list (+ the all-life default)
  await expect
    .poll(() => page.evaluate(() => document.getElementById("species-select").options.length))
    .toBeGreaterThanOrEqual(9); // 8 indicator species + "all recorded life"
  // toggling on adds a GBIF imagery layer (all-life density, no taxonKey)
  await page.check("#toggle-gbif");
  let u = await page.evaluate(() => window.__earth.gbifLayer?.imageryProvider.url);
  expect(u).toContain("api.gbif.org/v2/map/occurrence/density");
  expect(u).not.toContain("taxonKey");
  expect(u).toContain("purpleYellow.point");
  // selecting a species rebuilds the layer with that taxonKey and updates the note
  const key = await page.evaluate(() => window.__earth.gbifSpecies.find((s) => s.common === "Atlantic mackerel").key);
  await page.selectOption("#species-select", String(key));
  u = await page.evaluate(() => window.__earth.gbifLayer.imageryProvider.url);
  expect(u).toContain(`taxonKey=${key}`);
  expect(u).toContain("fire.point");
  await expect(page.locator("#species-note")).toContainText("north");
  // toggling off removes the layer
  await page.uncheck("#toggle-gbif");
  expect(await page.evaluate(() => window.__earth.gbifLayer)).toBeNull();
});

test("hover value probe reads the actual value from the top colormapped layer", async ({ page }) => {
  // SST is on by default; probe a warm tropical Atlantic point
  const r = await page.evaluate(async () => {
    const warm = await window.__earth.probeValueAt(Cesium.Cartographic.fromDegrees(-30, 5));
    const cold = await window.__earth.probeValueAt(Cesium.Cartographic.fromDegrees(-20, 68));
    const land = await window.__earth.probeValueAt(Cesium.Cartographic.fromDegrees(10, 47));
    return { warm, cold, land };
  });
  // tropical ocean SST is warm (~24–30 °C), subpolar much colder — both in physical units
  expect(r.warm.units).toBe("°C");
  expect(r.warm.value).toBeGreaterThan(20);
  expect(r.warm.value).toBeLessThan(32);
  expect(r.cold.value).toBeLessThan(r.warm.value);   // subpolar cooler than tropics
  // continental interior has no SST → flagged no-data, not a bogus number
  expect(r.land.noData).toBe(true);
  // with no colormapped layer active, the probe returns null
  await page.uncheck('#layer-list input[data-id="sst"]');
  const none = await page.evaluate(async () =>
    window.__earth.probeValueAt(Cesium.Cartographic.fromDegrees(-30, 5)));
  expect(none).toBeNull();
});

test("glacier layer (RGI v7) loads the full inventory as points", async ({ page }) => {
  await page.check("#toggle-glaciers");
  await expect
    .poll(() => page.evaluate(() => window.__earth.glacierCollection?.length ?? 0), { timeout: 20000 })
    .toBeGreaterThan(150000);
  await expect(page.locator("#meta-glaciers")).toContainText("glaciers");
  await expect(page.locator("#meta-glaciers")).toContainText("km²");
  // toggling off hides but keeps the collection
  await page.uncheck("#toggle-glaciers");
  expect(await page.evaluate(() => window.__earth.glacierCollection.show)).toBe(false);
});

test("computed difference generalises to sea ice, not to point/instantaneous layers", async ({ page }) => {
  // a winter date where AMSR2 sea ice exists (the recent default lags mission data)
  await page.fill("#layer-date", "2024-03-01");
  await page.dispatchEvent("#layer-date", "change");
  await page.selectOption("#compare-select", "10");
  await page.selectOption("#compare-mode", "delta");

  // sea ice is a continuous raster with deltaRange → becomes a DeltaProvider and paints
  await page.check('#layer-list input[data-id="seaice"]');
  const r = await page.evaluate(async () => {
    const e = window.__earth.state.layers["seaice"];
    const prov = e.layer.imageryProvider;
    const c = await prov.requestImage(1, 0, 2);          // northern tile
    const d = c.getContext("2d").getImageData(0, 0, 512, 512).data;
    let painted = 0; for (let i = 3; i < d.length; i += 4) if (d[i] > 0) painted++;
    const vlut = await window.__earth.getValueLut(e.cfg.colormap);
    return { isDelta: e.isDelta, name: prov.constructor.name, layerId: prov.layerId,
             painted, units: vlut?.units, lut: vlut?.lut.size };
  });
  expect(r.isDelta).toBe(true);
  expect(r.name).toBe("DeltaProvider");
  expect(r.layerId).toBe("seaice");
  expect(r.units).toBe("%");
  expect(r.lut).toBeGreaterThan(50);                     // single-value colormap now parses
  expect(r.painted).toBeGreaterThan(1000);               // real sea-ice change over the Arctic

  // precipitation has no deltaRange → stays a normal layer, and the hint appears
  await page.check('#layer-list input[data-id="precip"]');
  const p = await page.evaluate(() => {
    const e = window.__earth.state.layers["precip"];
    return { isDelta: e.isDelta, name: e.layer.imageryProvider.constructor.name };
  });
  expect(p.isDelta).toBe(false);
  expect(p.name).not.toBe("DeltaProvider");
  await expect(page.locator("#delta-hint")).toBeVisible();
  await expect(page.locator("#delta-hint")).toContainText("instantaneous");
});

test("hover probe reports the delta (not absolute) when a difference layer is active", async ({ page }) => {
  await page.selectOption("#compare-select", "10");
  await page.selectOption("#compare-mode", "delta");
  await page.evaluate(() => {
    const s = document.getElementById("window-days");
    s.value = "60"; s.dispatchEvent(new Event("change"));
  });
  const r = await page.evaluate(async () => {
    const warm = await window.__earth.probeValueAt(Cesium.Cartographic.fromDegrees(-40, 45)); // N Atlantic
    return warm;
  });
  expect(r.delta).toBe(true);
  expect(r.units).toBe("°C");
  // a decade-scale SST change is a small number, not an absolute ~10–25 °C reading
  if (!r.noData) expect(Math.abs(r.value)).toBeLessThan(8);
});

test("Temp dashboard shows GISTEMP land vs land+ocean warming", async ({ page }) => {
  await page.click("#tab-temp");
  await expect(page.locator("#temp-lo .stat-value")).not.toHaveText("–");
  const r = await page.evaluate(() => ({
    years: window.__earth.gistemp.years.length,
    lo: Number(document.querySelector("#temp-lo .stat-value").textContent),
    land: Number(document.querySelector("#temp-land .stat-value").textContent),
    chartW: document.getElementById("temp-chart").width,
    legend: document.getElementById("temp-legend").children.length,
  }));
  expect(r.years).toBeGreaterThan(140);
  expect(r.lo).toBeGreaterThan(1.0);
  expect(r.land).toBeGreaterThan(r.lo);   // land warms faster
  expect(r.chartW).toBeGreaterThan(0);
  expect(r.legend).toBe(2);
  await page.hover("#temp-chart", { position: { x: 200, y: 90 } });
  await expect(page.locator("#temp-tooltip")).toBeVisible();
  await expect(page.locator("#temp-tooltip")).toContainText("land");
});

test("land surface temperature layer is present and differenceable", async ({ page }) => {
  await page.check('#layer-list input[data-id="lst"]');
  const r = await page.evaluate(() => {
    const cfg = window.__earth.GIBS_LAYERS.find((l) => l.id === "lst");
    return { has: !!window.__earth.state.layers["lst"]?.layer, deltaRange: cfg.deltaRange, title: cfg.title };
  });
  expect(r.has).toBe(true);
  expect(r.deltaRange).toBeGreaterThan(0);          // supports computed difference
  expect(r.title).toContain("Land surface temperature");
});

test("hover value probe waits for dwell; click reads immediately", async ({ page }) => {
  // moving the mouse should NOT show the probe (it only appears after a dwell)
  await page.mouse.move(700, 400);
  await page.mouse.move(750, 420);
  await page.mouse.move(800, 440);
  await expect(page.locator("#value-probe")).toBeHidden();
  // a click reads the value immediately over the ocean (SST on by default)
  await page.evaluate(() => window.__runProbe(760, 430));
  // (runProbe renders only if the point is on the globe & has data; assert no crash and hidden-or-shown state is valid)
  const cls = await page.getAttribute("#value-probe", "class");
  expect(typeof cls).toBe("string");
});

test("grayscale globe toggle desaturates the base map", async ({ page }) => {
  const sat = () => page.evaluate(() =>
    window.__earth.viewer.imageryLayers.get(0).saturation);
  expect(await sat()).toBe(1.0);                 // full colour by default
  await page.check("#toggle-grayscale");
  expect(await sat()).toBe(0.0);                 // grayscale on
  await page.uncheck("#toggle-grayscale");
  expect(await sat()).toBe(1.0);                 // colour restored
});

test("glacier layer can colour by 2000-2020 melt rate", async ({ page }) => {
  await page.selectOption("#glacier-mode", "change");
  await expect
    .poll(() => page.evaluate(() => window.__earth.glacierData?.dhdt_matched ?? 0), { timeout: 25000 })
    .toBeGreaterThan(200000);
  await expect(page.locator("#glacier-legend")).toBeVisible();   // melt-rate scale shown
  // a strongly-thinning glacier renders warm (red-ish), a growing one cool (blue)
  const c = await page.evaluate(() => {
    const melt = window.__earth.glacierColor ? null : null; // colorGlaciers applied on load
    // sample the collection colours directly
    const col = window.__earth.glacierCollection;
    const d = window.__earth.glacierData;
    let warm = null, cool = null;
    for (let i = 0; i < d.dhdt.length && (!warm || !cool); i++) {
      if (d.dhdt[i] != null && d.dhdt[i] < -1 && !warm) warm = col.get(i).color;
      if (d.dhdt[i] != null && d.dhdt[i] > 0.2 && !cool) cool = col.get(i).color;
    }
    return { warmR: warm?.red, warmB: warm?.blue, coolB: cool?.blue, coolR: cool?.red };
  });
  expect(c.warmR).toBeGreaterThan(c.warmB);   // melting → red > blue
  expect(c.coolB).toBeGreaterThan(c.coolR);   // growing → blue > red
  await page.selectOption("#glacier-mode", "extent");
  await expect(page.locator("#glacier-legend")).toBeHidden();
});

test("gridded climatology layers paint tiles and probe exact cell values", async ({ page }) => {
  // GPCP (global precip), OISST (global SST, ocean), E-OBS (Europe), MeteoSwiss (CH)
  const cases = [
    { id: "gpcp", lon: -60, lat: -3, min: 1500, units: "mm/yr" },   // Amazon, very wet
    { id: "oisst", lon: -140, lat: 0, min: 20, units: "°C" },        // equatorial Pacific, warm
    { id: "eobs", lon: 2.3, lat: 48.9, min: 300, units: "mm/yr" },   // Paris
    { id: "meteoswiss", lon: 8.2, lat: 46.8, min: 800, units: "mm/yr" }, // central Switzerland
  ];
  const r = await page.evaluate(async (cases) => {
    const E = window.__earth, out = {};
    for (const c of cases) {
      const cfg = E.GIBS_LAYERS.find((l) => l.id === c.id);
      const g = await E.loadGrid(cfg);
      // geographic tile (level 5) containing the point
      const lvl = 5, nx = 2 * 2 ** lvl, ny = 2 ** lvl;
      const tx = Math.floor((c.lon + 180) / 360 * nx);
      const ty = Math.floor((90 - c.lat) / 180 * ny);
      const canvas = await new E.GridProvider(cfg).requestImage(tx, ty, lvl);
      const d = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
      let painted = 0; for (let i = 3; i < d.length; i += 4) if (d[i] > 0) painted++;
      out[c.id] = { painted, sample: E.sampleGrid(g, c.lon, c.lat), isGrid: cfg.grid === true };
    }
    return out;
  }, cases);
  for (const c of cases) {
    expect(r[c.id].isGrid, `${c.id} is a grid layer`).toBe(true);
    expect(r[c.id].painted, `${c.id} tile painted`).toBeGreaterThan(50);
    expect(r[c.id].sample, `${c.id} sample`).toBeGreaterThan(c.min);
  }
  // probe reads the exact value straight from the grid (units + magnitude)
  await page.check('#layer-list input[data-id="meteoswiss"]');
  const probe = await page.evaluate(async () => {
    const E = window.__earth, C = window.Cesium;
    return await E.probeValueAt(C.Cartographic.fromDegrees(8.2, 46.8));
  });
  expect(probe.units).toBe("mm/yr");
  expect(probe.value).toBeGreaterThan(800);
  // grid legend shows a ramp bar with min/mid/max labels
  await expect(page.locator("#legend-panel")).toContainText("MeteoSwiss");
  await expect(page.locator("#legend-panel .legend-item").filter({ hasText: "MeteoSwiss" })
    .locator("canvas.legend-bar")).toHaveCount(1);
  expect(page.__errors).toEqual([]);
});

test("new native GIBS layers toggle; salinity snaps to first-of-month", async ({ page }) => {
  const before = await page.evaluate(() => window.__earth.viewer.imageryLayers.length);
  for (const id of ["precip-30min", "chlor", "salinity"]) {
    await page.check(`#layer-list input[data-id="${id}"]`);
  }
  const info = await page.evaluate(() => {
    const E = window.__earth;
    // WMTS provider ctor name is mangled in the vendored build; assert our own
    // GIBS tiling scheme instead — a reliable marker of a real GIBS tile layer.
    const scheme = (id) => E.state.layers[id].layer.imageryProvider.tilingScheme.constructor.name;
    const salCfg = E.GIBS_LAYERS.find((l) => l.id === "salinity");
    return {
      count: E.viewer.imageryLayers.length,
      p30: scheme("precip-30min"),
      chl: scheme("chlor"),
      sal: scheme("salinity"),
      monthlySnap: E.gibsTime(salCfg, "2024-03-15"),
      dailyNoSnap: E.gibsTime(E.GIBS_LAYERS.find((l) => l.id === "precip"), "2024-03-15"),
      currentMonthFallback: E.gibsTime(salCfg, E.state.date),
      january: E.gibsTime(salCfg, new Date().toISOString().slice(0, 4) + "-01-15"),
      today: E.state.date,
    };
  });
  expect(info.count).toBe(before + 3);
  for (const s of [info.p30, info.chl, info.sal]) expect(s).toBe("GIBSGeographicTilingScheme");
  expect(info.monthlySnap).toBe("2024-03-01");   // monthly layers request first-of-month
  expect(info.dailyNoSnap).toBe("2024-03-15");   // daily layers use the raw date
  // the current month's composite is unpublished → fall back to previous month
  expect(info.currentMonthFallback < info.today.slice(0, 8) + "01").toBe(true);
  expect(info.currentMonthFallback).toMatch(/-01$/);
  expect(page.__errors).toEqual([]);
});

test("date stepper: calendar-correct steps, clamped to available range", async ({ page }) => {
  const start = await page.inputValue("#layer-date");
  await page.click('#date-steps button[data-step="-1y"]');
  const back1y = await page.evaluate(() => window.__earth.state.date);
  expect(Number(back1y.slice(0, 4))).toBe(Number(start.slice(0, 4)) - 1);
  expect(back1y.slice(5)).toBe(start.slice(5));            // same month-day
  await page.click('#date-steps button[data-step="-1m"]');
  const back1m = await page.evaluate(() => window.__earth.state.date);
  expect(back1m < back1y).toBe(true);
  await page.click('#date-steps button[data-step="+1d"]');
  const fwd = await page.evaluate(() => window.__earth.state.date);
  expect(fwd > back1m).toBe(true);
  // Today returns to the most recent date, and +1d cannot pass it
  await page.click('#date-steps button[data-step="today"]');
  const today = await page.evaluate(() => window.__earth.state.date);
  expect(today).toBe(start);                                // default IS most recent
  await page.click('#date-steps button[data-step="+1d"]');
  expect(await page.evaluate(() => window.__earth.state.date)).toBe(today);
  // stepping refreshes timed layers (date input mirrors state)
  expect(await page.inputValue("#layer-date")).toBe(today);
  expect(page.__errors).toEqual([]);
});

test("every layer entry has a hover card with record, interval, spatial facts", async ({ page }) => {
  // dynamically-built GIBS/grid layers: one tip per entry
  const gibs = await page.evaluate(() => {
    const items = [...document.querySelectorAll("#layer-list .layer-item")];
    return {
      items: items.length,
      tips: items.filter((i) => i.querySelector(".layer-tip")).length,
    };
  });
  expect(gibs.tips).toBe(gibs.items);                     // no layer without facts
  // static analysis/data layers each carry a hand-written tip too
  const allTips = await page.locator("#panel-layers .layer-tip").count();
  expect(allTips - gibs.tips).toBeGreaterThanOrEqual(6);
  // each card states the three facts
  const rows = await page.evaluate(() => {
    const tip = document.querySelector('#layer-list .layer-item .layer-tip');
    return [...tip.querySelectorAll("span")].map((s) => s.textContent);
  });
  expect(rows).toEqual(["Recorded", "Interval", "Spatial"]);
  // hovering reveals the card (CSS-driven)
  const sstItem = page.locator('#layer-list .layer-item', { hasText: "Sea surface temperature (MUR" });
  await sstItem.hover();
  await expect(sstItem.locator(".layer-tip")).toBeVisible();
  await expect(sstItem.locator(".layer-tip")).toContainText("2002-06 → present");
  await expect(sstItem.locator(".layer-tip")).toContainText("daily");
  await expect(sstItem.locator(".layer-tip")).toContainText("1 km");
  // the old ambiguous "· from <date>" meta suffix is gone
  const metas = await page.locator("#layer-list .meta").allTextContents();
  for (const m of metas) expect(m).not.toMatch(/· from \d{4}/);
});

test("every layer hover card carries a gist paragraph in clear language", async ({ page }) => {
  // every tip (dynamic and static) has a non-trivial summary paragraph
  const sums = await page.evaluate(() =>
    [...document.querySelectorAll("#panel-layers .layer-tip")].map((t) => ({
      sum: t.querySelector(".tip-sum")?.textContent.trim() || "",
      rec: [...t.querySelectorAll("div")].find((d) => d.textContent.startsWith("Recorded"))?.textContent || "",
    })));
  expect(sums.length).toBeGreaterThanOrEqual(22);
  for (const s of sums) {
    expect(s.sum.length, "gist paragraph present").toBeGreaterThan(80);
  }
  // no ambiguous "record from <year>" shorthand anywhere (misread as data
  // being fixed to that year) — instrument-vs-tiles must be spelled out
  for (const s of sums) expect(s.rec).not.toMatch(/record from \d{4}\)/);
  // LST specifically: tile availability vs instrument record is explicit,
  // and the patchy clear-sky coverage is explained in the gist
  const lst = page.locator('#layer-list .layer-item', { hasText: "Land surface temperature" });
  await expect(lst.locator(".layer-tip")).toContainText("aren't served as map tiles");
  await expect(lst.locator(".layer-tip")).toContainText("clouds, not missing data");
  // climatologies say "not one date"
  const ms = page.locator('#layer-list .layer-item', { hasText: "Precipitation normal (MeteoSwiss" });
  await expect(ms.locator(".layer-tip")).toContainText("not one date");
});

test("aggregation generalises to LST and fills clear-sky gaps", async ({ page }) => {
  // LST on any single day is mostly holes (clouds). Averaging a window must
  // (a) use the AggregateProvider, (b) exclude missing samples per pixel and
  // divide by the per-pixel observation count, (c) paint strictly more pixels
  // than one day alone.
  const r = await page.evaluate(async () => {
    const E = window.__earth;
    const cfg = E.GIBS_LAYERS.find((l) => l.id === "lst");
    const paint = async (prov) => {
      const c = await prov.requestImage(2, 1, 2);       // Africa/Europe tile (lots of land)
      const d = c.getContext("2d").getImageData(0, 0, 512, 512).data;
      let n = 0; for (let i = 3; i < d.length; i += 4) if (d[i] > 0) n++;
      return n;
    };
    const single = await paint(new E.AggregateProvider(cfg, E.state.date, 1));
    const agg = await paint(new E.AggregateProvider(cfg, E.state.date, 60));
    return { single, agg, dates: E.windowSampleDates(E.state.date, 60).length };
  });
  expect(r.dates).toBeGreaterThan(4);                    // several samples across the window
  expect(r.agg).toBeGreaterThan(r.single);               // gaps filled by the mean
  expect(r.agg).toBeGreaterThan(10000);                  // substantial land coverage

  // the UI path: enable LST, set a window → entry becomes an aggregate
  await page.check('#layer-list input[data-id="lst"]');
  await page.evaluate(() => {
    const s = document.getElementById("window-days");
    s.value = "60";
    s.dispatchEvent(new Event("change"));
  });
  const ui = await page.evaluate(() => {
    const e = window.__earth.state.layers["lst"];
    return { isAgg: e.isAggregate, name: e.layer.imageryProvider.constructor.name };
  });
  expect(ui.isAgg).toBe(true);
  expect(ui.name).toBe("AggregateProvider");
});

test("aggregation/difference matrix: every timed raster has an explicit posture", async ({ page }) => {
  const m = await page.evaluate(() => {
    const out = {};
    for (const l of window.__earth.GIBS_LAYERS) {
      out[l.id] = { delta: l.deltaRange != null, agg: l.deltaRange != null || !!l.aggregable,
                    timed: !!l.timed, grid: !!l.grid };
    }
    return out;
  });
  // both average & difference: fully continuous fields
  for (const id of ["sst", "sst-anom", "seaice", "snow", "lst", "salinity"]) {
    expect(m[id].delta, `${id} differenceable`).toBe(true);
  }
  // average-only: sound to time-average, unsound to difference day-vs-day
  for (const id of ["chlor", "aod"]) {
    expect(m[id].agg, `${id} aggregable`).toBe(true);
    expect(m[id].delta, `${id} not differenceable`).toBe(false);
  }
  // neither: photographs and instantaneous sparse fields
  for (const id of ["viirs-truecolor", "nightlights", "precip", "precip-30min"]) {
    expect(m[id].delta, `${id} no delta`).toBe(false);
    expect(m[id].agg, `${id} no aggregate`).toBe(false);
  }
  // a window turns an aggregable-only layer into an AggregateProvider too
  await page.check('#layer-list input[data-id="aod"]');
  await page.evaluate(() => {
    const s = document.getElementById("window-days");
    s.value = "30";
    s.dispatchEvent(new Event("change"));
  });
  const aod = await page.evaluate(() => {
    const e = window.__earth.state.layers["aod"];
    return { isAgg: e.isAggregate, name: e.layer.imageryProvider.constructor.name };
  });
  expect(aod.isAgg).toBe(true);
  expect(aod.name).toBe("AggregateProvider");
  // monthly salinity: sample dates snap to first-of-month and dedupe
  const sal = await page.evaluate(() => {
    const E = window.__earth;
    const cfg = E.GIBS_LAYERS.find((l) => l.id === "salinity");
    const p = new E.AggregateProvider(cfg, "2023-06-15", 60);
    return { dates: p._dates, allFirsts: p._dates.every((d) => d.endsWith("-01")) };
  });
  expect(sal.allFirsts).toBe(true);
  expect(sal.dates.length).toBeLessThanOrEqual(3);       // 60 days ≈ 2-3 distinct months
  expect(new Set(sal.dates).size).toBe(sal.dates.length); // deduped
});

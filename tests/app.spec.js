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

test("zoom buttons change camera height; trackpad pinch is registered", async ({ page }) => {
  const height = () =>
    page.evaluate(() => window.__earth.viewer.camera.positionCartographic.height);
  const h0 = await height();
  await page.click("#zoom-in");
  await expect.poll(height).toBeLessThan(h0 * 0.7);
  await page.click("#zoom-out");
  await expect.poll(height).toBeGreaterThan((await height()) - 1); // sanity: no throw
  // ctrl+wheel (browser encoding of a trackpad pinch) must be a registered zoom gesture
  const registered = await page.evaluate(() => {
    const types = window.__earth.viewer.scene.screenSpaceCameraController.zoomEventTypes;
    return types.some(
      (t) => t && t.eventType === Cesium.CameraEventType.WHEEL &&
             t.modifier === Cesium.KeyboardEventModifier.CTRL
    ) && types.includes(Cesium.CameraEventType.PINCH);
  });
  expect(registered).toBe(true);
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

test("catalog browser filters 241 datasets", async ({ page }) => {
  await page.click("#tab-catalog");
  await expect(page.locator("#catalog-count")).toHaveText("241 of 241 datasets");
  await page.fill("#catalog-search", "RAPID");
  const txt = await page.locator("#catalog-count").textContent();
  const n = Number(txt.split(" ")[0]);
  expect(n).toBeGreaterThan(0);
  expect(n).toBeLessThan(50);
  await page.fill("#catalog-search", "");
  await page.check("#filter-amoc");
  await expect(page.locator("#catalog-count")).toContainText("58 of 241");
});

test("every layer title is a clickable documentation link", async ({ page }) => {
  // GIBS layers: title itself links to the dataset docs, checkbox toggles separately
  const links = page.locator("#layer-list .layer-head a.title-link");
  await expect(links).toHaveCount(8);
  for (const href of await links.evaluateAll((as) => as.map((a) => a.href))) {
    expect(href).toMatch(/^https:\/\//);
  }
  await expect(links.first()).toHaveAttribute("target", "_blank");
  // data + analysis layers (SST ensemble, Climate TRACE, Argo, stations) too
  await expect(page.locator("#panel-layers .layer-head a.title-link")).toHaveCount(12);
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
  await expect(tip).toContainText("warmer");
  await bar.hover({ position: { x: box.width * 0.1, y: 5 } });
  await expect(tip).toContainText("cooler");
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
      providerIsDelta: e.layer.imageryProvider instanceof window.__earth.SSTDeltaProvider,
    };
  });
  expect(r.isDelta && r.noTwin && r.providerIsDelta).toBe(true);
  await expect(page.locator("#split-handle")).toBeHidden(); // no swipe in delta mode
  await expect(page.locator("#legend-panel")).toContainText("Δ SST");
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
  expect(r.main).toBe("SSTAggregateProvider");
  expect(r.cmp).toBe("SSTAggregateProvider");
  await expect(page.locator("#split-handle")).toBeVisible();

  // window applies even without comparison (single aggregated layer)
  await page.selectOption("#compare-select", "0");
  r = await page.evaluate(() => {
    const e = window.__earth.state.layers["sst"];
    return { isAgg: e.isAggregate, name: e.layer.imageryProvider.constructor.name, hasCmp: !!e.cmpLayer };
  });
  expect(r.isAgg).toBe(true);
  expect(r.name).toBe("SSTAggregateProvider");
  expect(r.hasCmp).toBe(false);

  // back to a single day → plain GIBS provider
  await setWindow(1);
  r = await page.evaluate(() => {
    const e = window.__earth.state.layers["sst"];
    return { isAgg: e.isAggregate, name: e.layer.imageryProvider.constructor.name };
  });
  expect(r.isAgg).toBe(false);
  expect(r.name).not.toBe("SSTAggregateProvider");
});

test("computed-difference hint appears only for non-SST layers in delta mode", async ({ page }) => {
  await page.selectOption("#compare-select", "10");
  await page.selectOption("#compare-mode", "delta");
  await expect(page.locator("#delta-hint")).toBeHidden(); // only SST on by default
  await page.check('#layer-list input[data-id="precip"]');
  await expect(page.locator("#delta-hint")).toBeVisible();
  await expect(page.locator("#delta-hint")).toContainText("sea surface temperature only");
  await page.selectOption("#compare-mode", "split");
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

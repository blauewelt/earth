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
    const omHosts = [
      ["https://api.open-meteo.com", "http://localhost:8083"],
      ["https://air-quality-api.open-meteo.com", "http://localhost:8084"],
      ["https://flood-api.open-meteo.com", "http://localhost:8085"],
      ["https://marine-api.open-meteo.com", "http://localhost:8086"],
      ["https://climate-api.open-meteo.com", "http://localhost:8087"],
    ];
    for (const [host, local] of omHosts) {
      await page.route(new RegExp(host.replace(/[.\/]/g, "\\$&") + "/.*"), async (route) => {
        try {
          const url = route.request().url().replace(host, local);
          const resp = await page.request.get(url);
          await route.fulfill({ response: resp });
        } catch {
          await route.abort().catch(() => {});
        }
      });
    }
  }
  page.__errors = [];
  page.on("pageerror", (e) => page.__errors.push(String(e)));
  await page.goto("/");
  await page.waitForFunction(() => window.__earth?.viewer, null, { timeout: 30000 });
});

/* Toasts auto-dismiss after 8 s and then remove themselves, so asserting on a
 * live `.toast` element only works if the assertion is the very next thing the
 * test does. Any test that checks a few other things first is timing the
 * animation, not the behaviour — and it fails exactly when the page is busiest,
 * which is precisely when a real regression would also hide.
 *
 * (2026-08-03: `tagline scenes` failed reproducibly this way. The sea-ice clamp
 * toast fired correctly every time; four intervening chip assertions on a page
 * still loading Arctic tiles took longer than the toast's life, so the element
 * was already gone — "element(s) not found", which reads like the feature is
 * missing.)
 *
 * Install this BEFORE the action, then assert on the log: it records every
 * toast the page has ever shown, so the assertion asks "did this message fire?"
 * instead of "is it still on screen right now?". */
async function recordToasts(page) {
  await page.evaluate(() => {
    if (window.__toastLog) return;
    const host = document.getElementById("toast-host");
    if (!host) return;
    window.__toastLog = [...host.querySelectorAll(".toast")].map((t) => t.textContent);
    new MutationObserver((recs) => {
      for (const r of recs) {
        for (const n of r.addedNodes) {
          if (n.nodeType === 1 && n.classList?.contains("toast")) {
            window.__toastLog.push(n.textContent);
          }
        }
      }
    }).observe(host, { childList: true });
  });
  return () => page.evaluate(() => (window.__toastLog ?? []).join(" ⏐ "));
}

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

  // wheel: one notch covers a big fraction of the height (fast zoom).
  // Positive deltaY (scroll down / pinch fingers together) zooms OUT.
  const before = await height();
  await page.evaluate(() =>
    window.__wheelZoom({ deltaY: 120, deltaMode: 0, ctrlKey: false, preventDefault() {} }));
  const afterWheel = await height();
  expect(afterWheel).toBeGreaterThan(before); // scroll down → zooms out

  // Negative deltaY (scroll up / pinch fingers apart) zooms IN.
  const beforeIn = await height();
  await page.evaluate(() =>
    window.__wheelZoom({ deltaY: -120, deltaMode: 0, ctrlKey: false, preventDefault() {} }));
  const afterIn = await height();
  expect(afterIn).toBeLessThan(beforeIn * 0.35); // ~0.85 of height per notch → big jump

  // touch pinch stays native
  const hasPinch = await page.evaluate(() =>
    window.__earth.viewer.scene.screenSpaceCameraController.zoomEventTypes
      .includes(Cesium.CameraEventType.PINCH));
  expect(hasPinch).toBe(true);

  // trackpad pinch (ctrlKey) with a small delta still zooms meaningfully.
  // Fingers apart (negative deltaY) zooms in.
  const b2 = await height();
  await page.evaluate(() =>
    window.__wheelZoom({ deltaY: -20, deltaMode: 0, ctrlKey: true, preventDefault() {} }));
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
  // The 12-month smoothing that makes the trend readable: variance of the
  // smoothed series must be well below the raw 10-day series (which swings
  // ±5 Sv around a ~1 Sv/decade signal), null gaps must stay gaps (no
  // bridges invented across array service breaks), and the mean must be
  // preserved — smoothing that shifts the level would be editorialising.
  const s = await page.evaluate(() => {
    const { t, moc, resolution_days } = window.__earth.rapid;
    const sm = window.__earth.movingMean(t, moc, resolution_days || 10, 365);
    const varOf = (a) => {
      const v = a.filter((x) => x != null);
      const m = v.reduce((x, y) => x + y, 0) / v.length;
      return { var: v.reduce((x, y) => x + (y - m) ** 2, 0) / v.length, mean: m };
    };
    const raw = varOf(moc), smo = varOf(sm);
    // contract on gaps, on a synthetic series: a null run longer than the
    // window must stay null in the output — no line across a dead array
    const synth = [...Array(30).fill(1), ...Array(60).fill(null), ...Array(30).fill(1)];
    const smSynth = window.__earth.movingMean(null, synth, 10, 365);
    const gapStaysGap = smSynth[60] === null && smSynth[45] === null && smSynth[0] !== null;
    return { rawVar: raw.var, smoVar: smo.var, rawMean: raw.mean, smoMean: smo.mean, gapStaysGap };
  });
  expect(s.smoVar).toBeLessThan(s.rawVar / 3);
  expect(Math.abs(s.smoMean - s.rawMean)).toBeLessThan(0.5);
  expect(s.gapStaysGap).toBe(true);
  // hover produces a tooltip carrying BOTH readings
  await page.hover("#amoc-chart", { position: { x: 150, y: 80 } });
  await expect(page.locator("#amoc-tooltip")).toBeVisible();
  await expect(page.locator("#amoc-tooltip")).toContainText("Sv");
  await expect(page.locator("#amoc-tooltip")).toContainText("12-mo mean");
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
  for (const href of await links.evaluateAll((as) => as.map((a) => a.href))) {
    expect(href).toMatch(/^https:\/\//);
  }
  await expect(links.first()).toHaveAttribute("target", "_blank");
  // The point is EVERY title, not a count: pinning the number here only means the
  // test fails the next time a layer is added, which is not a defect. Assert the
  // invariant instead — no layer in the panel may claim a source it can't cite —
  // with a floor so that "every" can't be satisfied by an empty panel.
  const unlinked = await page.locator("#panel-layers .layer-head").evaluateAll((hs) =>
    hs.filter((h) => !h.querySelector("a.title-link")).map((h) => h.textContent.trim()));
  expect(unlinked).toEqual([]);
  expect(await page.locator("#panel-layers .layer-head a.title-link").count())
    .toBeGreaterThan(25);
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
  // 274k glacier billboards plus a full delta tile set: this one genuinely
  // needs more than the 90 s default, and hitting that wall reported itself as
  // a mystery "Received: undefined" on whichever assertion happened to be
  // in flight rather than as "the test ran out of time".
  test.setTimeout(180000);
  await page.selectOption("#compare-select", "10");
  await page.selectOption("#compare-mode", "delta");
  await expect(page.locator("#delta-hint")).toBeHidden(); // SST alone is differenceable
  // precipitation has no deltaRange (day-vs-day rain is noise) → hint appears in delta mode
  await page.check('#layer-list input[data-id="precip"]');
  await expect(page.locator("#delta-hint")).toBeVisible();
  await expect(page.locator("#delta-hint")).toContainText("weather");
  await page.uncheck('#layer-list input[data-id="precip"]');
  await expect(page.locator("#delta-hint")).toBeHidden();
  // glaciers: single-snapshot note appears in delta AND side-by-side modes
  // (generous timeout: the 7.4 MB glacier snapshot loads while delta tiles
  // saturate the connection pool)
  // On from inside the page too, for the same reason it is switched off that
  // way below: `check()` waits for actionability the starved render loop
  // cannot grant while the snapshot is decoding.
  await page.evaluate(() => {
    const el = document.getElementById("toggle-glaciers");
    el.checked = true;
    el.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await expect(page.locator("#delta-hint")).toBeVisible({ timeout: 30000 });
  await expect(page.locator("#delta-hint")).toContainText("single inventory");
  await page.selectOption("#compare-mode", "split");
  await expect(page.locator("#delta-hint")).toBeVisible(); // still shown in side-by-side
  // Toggle off from inside the page: with 274k glacier billboards on a
  // software GL stack the render loop starves Playwright's actionability
  // checks, so a normal uncheck() can sit waiting for "stable" forever.
  await page.evaluate(() => {
    const el = document.getElementById("toggle-glaciers");
    el.checked = false;
    el.dispatchEvent(new Event("change", { bubbles: true }));
  });
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

test("biodiversity (GBIF) layer: broad taxonomic groups + indicator species", async ({ page }) => {
  // selector now offers all-life + broad-group optgroups (kingdoms, animal &
  // plant groups, humans) + indicator species
  const opts = await page.evaluate(() => {
    const sel = document.getElementById("species-select");
    return {
      total: sel.options.length,
      groups: [...sel.querySelectorAll("optgroup")].map((g) => g.label),
    };
  });
  expect(opts.total).toBeGreaterThanOrEqual(25);            // 8 kingdoms + 8 animal + 2 plant + human + 8 species + default
  expect(opts.groups).toContain("Kingdoms (all life splits into these)");
  expect(opts.groups).toContain("Major animal groups");
  expect(opts.groups).toContain("Climate-indicator species");
  // toggling on adds a GBIF imagery layer (all-life density, no taxonKey)
  await page.check("#toggle-gbif");
  let u = await page.evaluate(() => window.__earth.gbifLayer?.imageryProvider.url);
  expect(u).toContain("api.gbif.org/v2/map/occurrence/density");
  expect(u).not.toContain("taxonKey");
  expect(u).toContain("purpleYellow.point");
  // the default note explains the composition of "all recorded life" incl. humans
  await expect(page.locator("#species-note")).toContainText("kingdom");
  await expect(page.locator("#species-note")).toContainText("Homo sapiens");
  // picking a kingdom (Animalia, key 1) filters the map by that taxonKey
  await page.selectOption("#species-select", "1");
  u = await page.evaluate(() => window.__earth.gbifLayer.imageryProvider.url);
  expect(u).toContain("taxonKey=1");
  expect(u).toContain("fire.point");
  // the layer is all-time — the note says the date selector doesn't affect it
  await expect(page.locator("#species-note")).toContainText("all-time");
  // humans are present as their own pickable taxon, with a sparsity + privacy note
  const human = await page.evaluate(() =>
    [...document.querySelectorAll("#species-select option")].some((o) => /Homo sapiens/.test(o.textContent)));
  expect(human).toBe(true);
  await page.selectOption("#species-select", "2436436");   // Homo sapiens
  await expect(page.locator("#species-note")).toContainText("privacy");
  await expect(page.locator("#species-note")).toContainText("very sparse");
  await expect(page.locator("#species-note")).toContainText("not a date problem");
  // selecting an indicator species updates the note
  const key = await page.evaluate(() => window.__earth.gbifSpecies.find((s) => s.common === "Atlantic mackerel").key);
  await page.selectOption("#species-select", String(key));
  u = await page.evaluate(() => window.__earth.gbifLayer.imageryProvider.url);
  expect(u).toContain(`taxonKey=${key}`);
  await expect(page.locator("#species-note")).toContainText("poleward");
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

  // Precipitation is log-distributed: it never becomes a DeltaProvider, because
  // subtracting two rain fields is mostly palette quantization error. It takes the
  // other posture — a ×-fold RATIO — and the hint has to say which of the two
  // readings is on screen, since "red" means opposite arithmetic in each.
  await page.check('#layer-list input[data-id="precip"]');
  const p = await page.evaluate(() => {
    const e = window.__earth.state.layers["precip"];
    return { isDelta: e.isDelta, isRatio: e.isRatio,
             name: e.layer.imageryProvider.constructor.name };
  });
  expect(p.isDelta).toBe(false);
  expect(p.isRatio).toBe(true);
  expect(p.name).not.toBe("DeltaProvider");
  await expect(page.locator("#delta-hint")).toBeVisible();
  await expect(page.locator("#delta-hint")).toContainText("ratio");

  // True colour is a photograph — no colormap, so there is nothing to invert into
  // a difference at all. That is the third posture, and it must be stated rather
  // than left as an unexplained absence of change.
  await page.check('#layer-list input[data-id="viirs-truecolor"]');
  const t = await page.evaluate(() => {
    const e = window.__earth.state.layers["viirs-truecolor"];
    return { isDelta: e.isDelta, isRatio: e.isRatio };
  });
  expect(t.isDelta).toBe(false);
  expect(t.isRatio).toBe(false);
  await expect(page.locator("#delta-hint")).toContainText("shown as-is");
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

test("base globe auto-greys under colormapped data; overrides persist", async ({ page }) => {
  const sat = () => page.evaluate(() =>
    window.__earth.viewer.imageryLayers.get(0).saturation);
  // SST is on by default and colormapped → auto mode greys the base at open
  expect(await sat()).toBe(0.0);
  // remove the colormapped layer (stations are plain points) → colour returns
  await page.uncheck('#layer-list input[data-id="sst"]');
  expect(await sat()).toBe(1.0);
  // a photographic layer (no colormap to fight) keeps the colour base
  await page.check('#layer-list input[data-id="viirs-truecolor"]');
  expect(await sat()).toBe(1.0);
  // forced modes override auto in both directions
  await page.selectOption("#base-mode", "gray");
  expect(await sat()).toBe(0.0);
  await page.selectOption("#base-mode", "color");
  await page.check('#layer-list input[data-id="sst"]');
  expect(await sat()).toBe(1.0);                 // colour forced despite SST
  // the override persists for the next visit
  expect(await page.evaluate(() => localStorage.getItem("baseMode"))).toBe("color");
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
  const info = await page.evaluate(async () => {
    const E = window.__earth;
    // WMTS provider ctor name is mangled in the vendored build; assert our own
    // GIBS tiling scheme instead — a reliable marker of a real GIBS tile layer.
    const scheme = (id) => E.state.layers[id].layer.imageryProvider.tilingScheme.constructor.name;
    const salCfg = E.GIBS_LAYERS.find((l) => l.id === "salinity");
    // Wait for the published time domain instead of racing the fetch addLayer()
    // just kicked off — the snapped answers below only mean anything once the
    // app knows what the archive actually serves.
    const dom = await E.loadGibsDomain(salCfg);
    return {
      count: E.viewer.imageryLayers.length,
      p30: scheme("precip-30min"),
      chl: scheme("chlor"),
      sal: scheme("salinity"),
      monthlyRequest: E.gibsTimeStatic(salCfg, "2024-03-15"),
      monthlySnap: E.gibsTime(salCfg, "2024-03-15"),
      dailyNoSnap: E.gibsTime(E.GIBS_LAYERS.find((l) => l.id === "precip"), "2024-03-15"),
      currentMonthFallback: E.gibsTime(salCfg, E.state.date),
      served: (dom || []).map((iv) => [iv.s, iv.e]),
      today: E.state.date,
    };
  });
  expect(info.count).toBe(before + 3);
  for (const s of [info.p30, info.chl, info.sal]) expect(s).toBe("GIBSGeographicTilingScheme");
  expect(info.monthlyRequest).toBe("2024-03-01");   // monthly layers request first-of-month
  expect(info.dailyNoSnap).toBe("2024-03-15");      // daily layers use the raw date
  // ...and the request is then snapped onto what SMAP really serves. That
  // archive has holes — as measured, 2022-09 and the whole of 2024 are missing —
  // so first-of-March-2024 resolves to an earlier month. Derive the expectation
  // from the domain the app just measured rather than pinning a month here: a
  // pinned month is a guess about someone else's archive, and it rots the moment
  // NASA backfills.
  expect(info.served.length).toBeGreaterThan(0);
  const covers = (d) => info.served.some(([s, e]) => s <= d && d <= e);
  expect(covers(info.monthlySnap)).toBe(true);
  expect(info.monthlySnap <= info.monthlyRequest).toBe(true);
  expect(info.monthlySnap).toMatch(/-01$/);
  // the current month's composite is unpublished → fall back to an earlier month
  expect(info.currentMonthFallback < info.today.slice(0, 8) + "01").toBe(true);
  expect(info.currentMonthFallback).toMatch(/-01$/);
  expect(covers(info.currentMonthFallback)).toBe(true);
  expect(page.__errors).toEqual([]);
});

test("catch-all colormap bins probe as bounds, not invented midpoints", async ({ page }) => {
  // SMAP salinity's palette runs 30–40 PSU in 0.04-wide bins, but its first
  // entry is one catch-all [0,30) and its last [40,+INF). The probe used to
  // print the catch-all's midpoint — a flat "15.0 PSU" across the whole
  // Baltic (true value ~7) right next to honest 31s — an invented number the
  // reader can't tell from a measured one. Now a capped bin answers with its
  // bound: "< 30".
  await page.check('#layer-list input[data-id="salinity"]');
  const r = await page.evaluate(async () => {
    const E = window.__earth;
    const cfg = E.GIBS_LAYERS.find((l) => l.id === "salinity");
    await E.loadGibsDomain(cfg);          // so the probe date snaps to a served month
    const vlut = await E.getValueLut(cfg.colormap);
    // salinity sits above SST, so an opaque salinity pixel wins the probe
    const baltic = await E.probeValueAt(Cesium.Cartographic.fromDegrees(20, 57.3));
    const atlantic = await E.probeValueAt(Cesium.Cartographic.fromDegrees(-30, 40));
    return { caps: [...vlut.caps.values()], units: vlut.units, baltic, atlantic };
  });
  // both ends of the palette are recognised as caps, from the real colormap
  expect(r.caps).toContainEqual({ sign: "<", bound: 30 });
  expect(r.caps).toContainEqual({ sign: "≥", bound: 40 });
  // Baltic proper is brackish → the catch-all bin → reads as a bound...
  expect(r.baltic.title).toContain("salinity");
  expect(r.baltic.cap).toEqual({ sign: "<", bound: 30 });
  // ...while the numeric value stays the midpoint for delta/mean arithmetic
  expect(r.baltic.value).toBe(15);
  // mid-Atlantic is squarely on the scale: a real number, no cap
  expect(r.atlantic.value).toBeGreaterThan(30);
  expect(r.atlantic.value).toBeLessThan(40);
  expect(r.atlantic.cap).toBeFalsy();
  expect(r.units).toBe("PSU");
  expect(page.__errors).toEqual([]);
});

test("the probe draws the source cell it read on the globe", async ({ page }) => {
  // A tap's read-out floats OFFSET from the finger, so nothing said which
  // pixel the number came from. Every probe now returns the source cell's
  // geographic footprint and the globe outlines it, plus a ring at the tap.
  const r = await page.evaluate(async () => {
    const E = window.__earth;
    // pure geometry: the cell is exactly one 512th of its tile
    const span = (0.5625 / 2 ** 5) * 512, cs = span / 512;
    const g = E.probeCellBounds(5, 22, 3, 100, 200);
    const geom = {
      ok: Math.abs(g.west - (-180 + 22 * span + 100 * cs)) < 1e-9 &&
          Math.abs(g.north - (90 - 3 * span - 200 * cs)) < 1e-9 &&
          Math.abs(g.east - g.west - cs) < 1e-9 &&
          Math.abs(g.north - g.south - cs) < 1e-9,
    };
    // behavioural: probe the canvas centre (always on the globe at the home view)
    const canvas = E.viewer.scene.canvas;
    await window.__runProbe(canvas.clientWidth / 2, canvas.clientHeight / 2);
    const m = E.probeMark;
    const rect = m?.fill.rectangle.coordinates.getValue(E.viewer.clock.currentTime);
    const dotPos = m?.dot.position.getValue(E.viewer.clock.currentTime);
    const dotCarto = dotPos && Cesium.Cartographic.fromCartesian(dotPos);
    return {
      geom,
      dot: m?.dot.show, fill: m?.fill.show, edge: m?.edge.show,
      rect: rect && {
        w: Cesium.Math.toDegrees(rect.west), s: Cesium.Math.toDegrees(rect.south),
        e: Cesium.Math.toDegrees(rect.east), n: Cesium.Math.toDegrees(rect.north),
      },
      dotLon: dotCarto && Cesium.Math.toDegrees(dotCarto.longitude),
      dotLat: dotCarto && Cesium.Math.toDegrees(dotCarto.latitude),
      probeShown: !document.getElementById("value-probe").classList.contains("hidden"),
    };
  });
  expect(r.geom.ok).toBe(true);
  expect(r.probeShown).toBe(true);
  expect(r.dot).toBe(true);
  expect(r.fill).toBe(true);
  expect(r.edge).toBe(true);
  // the read-out anchors CLEAR of the tap: a finger-sized gap, never on top
  // of the mark it describes (that is how it shipped for mice, and on a phone
  // the box sat on the very pixel it was reporting)
  const clear = await page.evaluate(() => {
    const canvas = window.__earth.viewer.scene.canvas;
    const cx = canvas.clientWidth / 2, cy = canvas.clientHeight / 2;
    const c = canvas.getBoundingClientRect();
    const b = document.getElementById("value-probe").getBoundingClientRect();
    const inside = cx + c.left >= b.left - 8 && cx + c.left <= b.right + 8 &&
                   cy + c.top >= b.top - 8 && cy + c.top <= b.bottom + 8;
    return { inside, w: b.width, h: b.height };
  });
  expect(clear.w).toBeGreaterThan(0);
  expect(clear.inside).toBe(false);
  // the outlined cell is a real source pixel: tiny, and it contains the tap
  expect(r.rect.e - r.rect.w).toBeGreaterThan(0);
  expect(r.rect.e - r.rect.w).toBeLessThan(1);
  expect(r.dotLon).toBeGreaterThanOrEqual(r.rect.w - 1e-6);
  expect(r.dotLon).toBeLessThanOrEqual(r.rect.e + 1e-6);
  expect(r.dotLat).toBeGreaterThanOrEqual(r.rect.s - 1e-6);
  expect(r.dotLat).toBeLessThanOrEqual(r.rect.n + 1e-6);
  // moving the pointer hides the read-out AND the marks together
  await page.mouse.move(400, 300);
  await page.mouse.move(420, 310);
  await expect(page.locator("#value-probe")).toBeHidden();
  const after = await page.evaluate(() => {
    const m = window.__earth.probeMark;
    return { dot: m.dot.show, fill: m.fill.show, edge: m.edge.show };
  });
  expect(after).toEqual({ dot: false, fill: false, edge: false });
  expect(page.__errors).toEqual([]);
});

test("a mark hidden under the pixel card rotates back into view", async ({ page }) => {
  // The pixel card covers a big slab of the globe (most of it, on a phone).
  // Opening it for a point that sits underneath it must rotate the globe so
  // the marked pixel is actually visible next to the card describing it.
  const r = await page.evaluate(async () => {
    const E = window.__earth;
    const scene = E.viewer.scene;
    const canvas = scene.canvas;
    const pick = (x, y) =>
      E.viewer.camera.pickEllipsoid(new Cesium.Cartesian2(x, y), scene.globe.ellipsoid);
    // zoom to where the globe fills the frame — at the far home view the
    // card's screen region can hang off the limb and there is nothing to mark
    E.viewer.camera.setView({
      destination: Cesium.Cartesian3.fromDegrees(-30, 40, 2.5e6),
    });
    // a globe point in the card's screen region (card: top-right, 300px wide)
    const cart = pick(canvas.clientWidth - 120, 140);
    if (!cart) return { skip: true };
    const carto = Cesium.Cartographic.fromCartesian(cart);
    const camBefore = E.viewer.camera.positionCartographic.clone();
    const cardP = E.showPixelState(carto);          // marks + rotates immediately
    await new Promise((res) => setTimeout(res, 1500));   // let the 0.6s flight land
    const st = Cesium.SceneTransforms;
    const toWin = (st.worldToWindowCoordinates || st.wgs84ToWindowCoordinates).bind(st);
    const m = E.probeMark;
    const pos = m.dot.position.getValue(E.viewer.clock.currentTime);
    const w = toWin(scene, pos);
    const cr = canvas.getBoundingClientRect();
    const card = document.getElementById("pixel-card").getBoundingClientRect();
    const camAfter = E.viewer.camera.positionCartographic;
    cardP.catch(() => {});                          // don't leak the card's own promise
    return {
      dot: m.dot.show,
      cardShown: !document.getElementById("pixel-card").classList.contains("hidden"),
      moved: Math.abs(camAfter.longitude - camBefore.longitude) +
             Math.abs(camAfter.latitude - camBefore.latitude) > 1e-4,
      onCanvas: !!w && w.x >= 0 && w.y >= 0 && w.x <= cr.width && w.y <= cr.height,
      underCard: !!w &&
        w.x + cr.left >= card.left && w.x + cr.left <= card.right &&
        w.y + cr.top >= card.top && w.y + cr.top <= card.bottom,
      // a second call must be a no-op: the mark is already visible
      movesAgain: E.ensureMarkVisible(),
    };
  });
  if (r.skip) test.skip();
  expect(r.cardShown).toBe(true);
  expect(r.dot).toBe(true);
  expect(r.moved).toBe(true);        // the camera rotated...
  expect(r.onCanvas).toBe(true);     // ...and the mark is on screen...
  expect(r.underCard).toBe(false);   // ...not under the card any more
  expect(r.movesAgain).toBe(false);
  // closing the card clears the marks it owned
  await page.click("#pixel-card .px-close");
  const after = await page.evaluate(() => {
    const m = window.__earth.probeMark;
    return { dot: m.dot.show, fill: m.fill.show, edge: m.edge.show };
  });
  expect(after).toEqual({ dot: false, fill: false, edge: false });
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
  for (const id of ["sst", "sst-anom", "seaice", "snow", "lst", "salinity",
                    "ndvi", "grace", "ssh-anom", "ceres"]) {
    expect(m[id].delta, `${id} differenceable`).toBe(true);
  }
  // average-only: sound to time-average, unsound to difference day-vs-day
  // (precip: daily-mean rates average like GPCP does; day-vs-day is weather;
  // soil moisture: day deltas compare swath coverage, not soil)
  for (const id of ["chlor", "aod", "precip", "soilmoisture"]) {
    expect(m[id].agg, `${id} aggregable`).toBe(true);
    expect(m[id].delta, `${id} not differenceable`).toBe(false);
  }
  // neither: photographs and instantaneous snapshots
  for (const id of ["viirs-truecolor", "nightlights", "precip-30min"]) {
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

test("non-aggregatable layers are hidden + warned under an active window", async ({ page }) => {
  const setWin = (v) => page.evaluate((val) => {
    const s = document.getElementById("window-days"); s.value = String(val);
    s.dispatchEvent(new Event("change"));
  }, v);
  // a half-hourly snapshot cannot be time-averaged; single day → it renders
  await page.check('#layer-list input[data-id="precip-30min"]');
  let r = await page.evaluate(() => window.__earth.state.layers["precip-30min"]);
  expect(!!r.layer).toBe(true);
  // a window suppresses it entirely rather than showing a misleading single day
  await setWin(90);
  r = await page.evaluate(() => {
    const e = window.__earth.state.layers["precip-30min"];
    return { has: !!e.layer, suppressed: !!e.suppressed };
  });
  expect(r.has).toBe(false);
  expect(r.suppressed).toBe(true);
  await expect(page.locator("#delta-hint")).toBeVisible();
  await expect(page.locator("#delta-hint")).toContainText("Precipitation rate");
  await expect(page.locator("#delta-hint")).toContainText("hidden while");
  // …and the hint points at the alternative that DOES average
  await expect(page.locator("#delta-hint")).toContainText("daily");
  // an aggregatable layer under the same window still shows (as an average)
  const sst = await page.evaluate(() => {
    const e = window.__earth.state.layers["sst"];
    return { has: !!e.layer, agg: e.isAggregate };
  });
  expect(sst.has && sst.agg).toBe(true);
  // returning to a single day restores the suppressed layer and clears the note
  await setWin(1);
  r = await page.evaluate(() => {
    const e = window.__earth.state.layers["precip-30min"];
    return { has: !!e.layer, suppressed: !!e.suppressed };
  });
  expect(r.has).toBe(true);
  expect(r.suppressed).toBe(false);
});

test("daily precip aggregates (dry = zero); 30-min layer steps through the day", async ({ page }) => {
  // -- daily: a window turns it into a real windowed mean, not a suppression
  await page.check('#layer-list input[data-id="precip"]');
  await page.evaluate(() => {
    const s = document.getElementById("window-days"); s.value = "5";
    s.dispatchEvent(new Event("change"));
  });
  const daily = await page.evaluate(() => {
    const e = window.__earth.state.layers["precip"];
    return { agg: e.isAggregate, sup: !!e.suppressed,
             name: e.layer.imageryProvider.constructor.name,
             zero: !!e.cfg.transparentZero };
  });
  expect(daily.sup).toBe(false);
  expect(daily.agg).toBe(true);
  expect(daily.name).toBe("AggregateProvider");
  expect(daily.zero).toBe(true);  // transparent = "no rain", counted as 0 in the mean
  // no ⚠ on its chip — it is genuinely being drawn as an average
  const chip = page.locator("#active-layers .chip", { hasText: "GPM IMERG V07)" });
  await expect(chip).not.toHaveClass(/chip-warn/);
  await page.evaluate(() => {
    const s = document.getElementById("window-days"); s.value = "1";
    s.dispatchEvent(new Event("change"));
  });

  // -- 30-min: the time row appears with the layer and steps in half-hours
  // Toggle in-page rather than via check/uncheck: with two IMERG rasters
  // streaming tiles the software-GL render loop starves Playwright's
  // actionability checks, and even a plain locator read can outlast the
  // timeout (§4 in CLAUDE.md — the same reason the glacier layer is toggled
  // this way).
  const toggle = (id, on) => page.evaluate(([i, v]) => {
    const el = document.querySelector(`#layer-list input[data-id="${i}"]`);
    if (el.checked !== v) { el.checked = v; el.dispatchEvent(new Event("change", { bubbles: true })); }
  }, [id, on]);
  const row = page.locator("#time-steps");
  await expect(row).toHaveClass(/hidden/);
  await toggle("precip-30min", true);
  await expect(row).not.toHaveClass(/hidden/);
  await expect(page.locator("#time-value")).toHaveText("00:00 UTC");
  await page.click('#time-steps button[data-tstep="+30"]');
  await expect(page.locator("#time-value")).toHaveText("00:30 UTC");
  // the GIBS TIME the layer actually requests carries the half-hour
  const t = await page.evaluate(() => {
    const E = window.__earth;
    const cfg = E.GIBS_LAYERS.find((l) => l.id === "precip-30min");
    return { sub: E.gibsTime(cfg, E.state.date),
             daily: E.gibsTime(E.GIBS_LAYERS.find((l) => l.id === "precip"), E.state.date) };
  });
  expect(t.sub).toMatch(/T00:30:00Z$/);
  expect(t.daily).not.toContain("T");  // daily layers are untouched by the stepper
  // stepping back across midnight rolls the date
  const dateBefore = await page.evaluate(() => document.getElementById("layer-date").value);
  await page.click('#time-steps button[data-tstep="-30"]');
  await page.click('#time-steps button[data-tstep="-30"]');
  await expect(page.locator("#time-value")).toHaveText("23:30 UTC");
  const after = await page.evaluate(() => window.__earth.state.date);
  expect(after < dateBefore).toBe(true);
  // The input must follow the state, or the date the user reads and the date
  // the tiles are fetched for have silently diverged.
  expect(await page.evaluate(() => document.getElementById("layer-date").value)).toBe(after);
  // switching the layer off hides the row again
  await toggle("precip-30min", false);
  await expect(row).toHaveClass(/hidden/);
});

test("enabling a date-independent layer fires an animated warning toast", async ({ page }) => {
  // a climatology grid has no per-date data → toast on enable
  await page.check('#layer-list input[data-id="gpcp"]');
  const toast = page.locator("#toast-host .toast").first();
  await expect(toast).toBeVisible();
  await expect(toast).toContainText("climatology");
  await expect(toast).toContainText("date selector doesn't change it");
  // it is actually animated — the entrance + shake keyframes are defined and
  // the element carries a non-zero animation
  const anim = await page.evaluate(() => {
    let names = new Set();
    for (const ss of document.styleSheets) {
      let rules; try { rules = ss.cssRules; } catch { continue; }
      for (const r of rules || []) if (r.type === CSSRule.KEYFRAMES_RULE) names.add(r.name);
    }
    const el = document.querySelector("#toast-host .toast");
    return { keyframes: [...names], dur: el && getComputedStyle(el).animationDuration };
  });
  expect(anim.keyframes).toContain("toast-in");
  expect(anim.keyframes).toContain("toast-shake");
  expect(anim.dur).not.toBe("0s");
  // dismissible — close via an in-page click (no actionability wait, so the
  // 8 s auto-dismiss can't race the test)
  await page.evaluate(() => document.querySelector("#toast-host .toast .toast-close").click());
  await expect(page.locator("#toast-host .toast")).toHaveCount(0);

  // Dismissing releases the de-dupe key rather than stranding it: switch the
  // same layer off and on and the message says itself again. The key exists to
  // stop two copies sharing the screen, not to remember what has been said —
  // a stranded key makes a message silently unsayable for the whole session.
  const flip = (id, v) => page.evaluate(([i, on]) => {
    const el = document.querySelector(`#layer-list input[data-id="${i}"]`);
    el.checked = on;
    el.dispatchEvent(new Event("change", { bubbles: true }));
  }, [id, v]);
  await flip("gpcp", false);
  await flip("gpcp", true);
  await expect(page.locator("#toast-host .toast")).toContainText("climatology");
  await page.evaluate(() => document.querySelector("#toast-host .toast .toast-close").click());
  await expect(page.locator("#toast-host .toast")).toHaveCount(0);
  await flip("gpcp", false);

  // date-DRIVEN layers must NOT toast (pure logic, no timing)
  expect(await page.evaluate(() => window.__earth.datelessToast("precip"))).toBeNull();
  expect(await page.evaluate(() => window.__earth.datelessToast("sst"))).toBeNull();
  await page.check('#layer-list input[data-id="precip"]');
  await page.waitForTimeout(300);
  await expect(page.locator("#toast-host .toast")).toHaveCount(0);

  // data/point + all-time layers toast with a tailored message
  expect(await page.evaluate(() => window.__earth.datelessToast("gbif"))).toContain("all-time");
  expect(await page.evaluate(() => window.__earth.datelessToast("glaciers"))).toContain("single inventory");
  await page.check("#toggle-glaciers");
  await expect(page.locator("#toast-host .toast")).toContainText("single inventory");
});

test("active-layer chips list what's on and switch it off from anywhere", async ({ page }) => {
  test.setTimeout(150000); // several layer loads + two tab switches
  const chips = page.locator("#active-layers .chip:not(.chip-clear)");
  const labels = () =>
    page.locator("#active-layers .chip-label").allTextContents();

  // The two default-on layers each get a chip: a raster and a point layer,
  // proving the chips don't care which machinery draws them.
  const has = async (frag) => (await labels()).some((s) => s.includes(frag));
  await expect(chips).toHaveCount(2);
  expect(await has("Sea surface temperature")).toBe(true);
  expect(await has("Monitoring stations")).toBe(true);

  await page.check("#toggle-climatetrace");
  await expect(chips).toHaveCount(3);
  expect(await has("Facility emissions")).toBe(true);

  // The × turns the layer off for real — not just visually.
  await page
    .locator("#active-layers .chip", { hasText: "Sea surface temperature" })
    .locator(".chip-x")
    .click();
  await expect(chips).toHaveCount(2);
  expect(await has("Sea surface temperature")).toBe(false);
  expect(await page.evaluate(() => !!window.__earth.state.layers["sst"]?.layer)).toBe(false);
  // and the sidebar checkbox followed along
  await expect(page.locator('#layer-list input[data-id="sst"]')).not.toBeChecked();

  // The whole point: reachable from a tab where the layer list isn't rendered.
  await page.click("#tab-catalog");
  await expect(page.locator("#panel-layers")).toBeHidden();
  await expect(page.locator("#active-layers")).toBeVisible();
  await page
    .locator("#active-layers .chip", { hasText: "Facility emissions" })
    .locator(".chip-x")
    .click();
  await expect(chips).toHaveCount(1);
  expect(await page.evaluate(
    () => window.__earth.pointLayers.climatetrace?.collection.show ?? false)).toBe(false);
  // one layer left → nothing to "clear all"
  await expect(page.locator("#active-layers .chip-clear")).toHaveCount(0);

  // Clicking the label jumps back to the layer's row and highlights it.
  // Read the highlight in the same tick as the click: it self-clears after
  // ~1.4 s, which a round-trip can outlast on a slow machine.
  await page.click("#tab-catalog");
  const flashed = await page.evaluate(() => {
    document.querySelector("#active-layers .chip-label").click();
    return document.getElementById("toggle-stations")
      .closest(".layer-item").classList.contains("flash");
  });
  expect(flashed).toBe(true);
  await expect(page.locator("#panel-layers")).toBeVisible();

  // "Clear all" appears past one layer and empties the globe.
  await page.check('#layer-list input[data-id="sst"]');
  await page.check('#layer-list input[data-id="precip"]');
  await expect(page.locator("#active-layers .chip-clear")).toContainText("Clear all 3");
  await page.locator("#active-layers .chip-clear .chip-label").click();
  await expect(page.locator("#active-layers")).toHaveClass(/hidden/);
  await expect(chips).toHaveCount(0);
  expect(await page.evaluate(() => Object.values(window.__earth.state.layers).some((e) => !!e.layer)))
    .toBe(false);
});

test("a suppressed layer is flagged on its chip rather than silently listed", async ({ page }) => {
  // The 30-min layer: a half-hourly snapshot can't be averaged, so a window
  // suppresses it (daily precip, by contrast, aggregates and must NOT warn).
  await page.check('#layer-list input[data-id="precip-30min"]');
  const chip = page.locator("#active-layers .chip", { hasText: "Precipitation rate" });
  await expect(chip).not.toHaveClass(/chip-warn/);
  // a window it can't honour → still checked, but marked as not being drawn
  await page.evaluate(() => {
    const s = document.getElementById("window-days"); s.value = "90";
    s.dispatchEvent(new Event("change"));
  });
  await expect(chip).toHaveClass(/chip-warn/);
  await expect(chip.locator(".chip-label")).toContainText("⚠");
});

test("log fields compare as ×-fold ratios of window means", async ({ page }) => {
  test.setTimeout(150000); // a computed-comparison render + a 2-tile probe
  await page.check('#layer-list input[data-id="precip"]');
  await page.selectOption("#compare-select", "10");
  await page.selectOption("#compare-mode", "delta");
  const r = await page.evaluate(() => {
    const e = window.__earth.state.layers["precip"];
    const sst = window.__earth.state.layers["sst"];
    return { isRatio: e.isRatio, isDelta: e.isDelta,
             name: e.layer.imageryProvider.constructor.name,
             sstDelta: sst.isDelta };  // SST keeps a true difference beside it
  });
  expect(r.isRatio).toBe(true);
  expect(r.isDelta).toBe(false);
  expect(r.name).toBe("RatioProvider");
  expect(r.sstDelta).toBe(true);
  // the hint explains the multiplicative reading, the legend shows the × axis
  await expect(page.locator("#delta-hint")).toContainText("ratio");
  await expect(page.locator("#legend-panel")).toContainText("ratio of");
  await expect(page.locator("#legend-panel")).toContainText("×8 more");
  // a rendered tile paints real change: rain moves in 10 years, so both red
  // (wetter) and blue (drier) pixels must appear
  const painted = await page.evaluate(async () => {
    const E = window.__earth;
    const cfg = E.GIBS_LAYERS.find((l) => l.id === "precip");
    const p = new E.RatioProvider(cfg, E.state.date, E.compareDate(), 1);
    const canvas = await p.requestImage(1, 0, 1);
    const d = canvas.getContext("2d").getImageData(0, 0, 512, 512).data;
    let red = 0, blue = 0;
    for (let i = 0; i < d.length; i += 4) {
      if (d[i + 3] === 0) continue;
      if (d[i] > d[i + 2]) red++; else blue++;
    }
    return { red, blue };
  });
  expect(painted.red).toBeGreaterThan(500);
  expect(painted.blue).toBeGreaterThan(500);
  // the probe reports a fold change (or honestly no data if the pixel is dry)
  const probe = await page.evaluate(() =>
    window.__earth.probeValueAt(Cesium.Cartographic.fromDegrees(-60, -5))); // Amazon
  expect(probe.ratio).toBe(true);
  if (!probe.noData) expect(probe.value).toBeGreaterThan(0);
});

test("window presets jump the slider; the explainer folds away", async ({ page }) => {
  // presets drive the same path as dragging the slider
  await page.click('#window-presets button[data-win="30"]');
  await expect(page.locator("#window-value")).toHaveText("past 30 days");
  await expect(page.locator('#window-presets button[data-win="30"]')).toHaveClass(/active/);
  let s = await page.evaluate(() => ({
    days: window.__earth.state.windowDays,
    sstAgg: window.__earth.state.layers["sst"].isAggregate,
  }));
  expect(s.days).toBe(30);
  expect(s.sstAgg).toBe(true);
  // 1d returns to a plain single-day layer
  await page.click('#window-presets button[data-win="1"]');
  await expect(page.locator("#window-value")).toHaveText("single day");
  await expect(page.locator('#window-presets button[data-win="1"]')).toHaveClass(/active/);
  s = await page.evaluate(() => ({
    days: window.__earth.state.windowDays,
    sstAgg: window.__earth.state.layers["sst"].isAggregate,
  }));
  expect(s.days).toBe(1);
  expect(s.sstAgg).toBe(false);
  // a hand-dragged odd value un-highlights every preset
  await page.evaluate(() => {
    const sl = document.getElementById("window-days"); sl.value = "42";
    sl.dispatchEvent(new Event("change"));
  });
  await expect(page.locator("#window-presets button.active")).toHaveCount(0);

  // the Compare/Aggregate explainer is collapsed by default and opens on click
  const details = page.locator("#how-compare");
  expect(await details.evaluate((el) => el.open)).toBe(false);
  await page.click("#how-compare summary");
  expect(await details.evaluate((el) => el.open)).toBe(true);
  await expect(details).toContainText("ratio");
});

test("the intro guide greets first visits, then remembers being dismissed", async ({ page }) => {
  // open by default for a new visitor, and it actually documents the view
  const intro = page.locator("#intro-guide");
  expect(await intro.evaluate((el) => el.open)).toBe(true);
  await expect(intro).toContainText("control room");
  await expect(intro).toContainText("Compare");
  await expect(intro).toContainText("Aggregate");
  await expect(intro).toContainText("Everything we know");
  await expect(intro).toContainText("The other tabs");
  // dismissing it persists across reloads — returning users get controls on top
  await page.click("#intro-guide summary");
  expect(await intro.evaluate((el) => el.open)).toBe(false);
  await page.reload();
  await page.waitForFunction(() => window.__earth?.viewer, null, { timeout: 30000 });
  expect(await page.locator("#intro-guide").evaluate((el) => el.open)).toBe(false);
});

test("pixel inspector composes a point's full state on click", async ({ page }) => {
  // ~9 raster tiles + 4 grids + live weather, twice — and every row now also
  // resolves its own observation time, so the budget is no longer marginal.
  test.setTimeout(240000);
  // "Everything we know" is a layer-list entry, off by default: with SST on,
  // a plain click reads the SST value, not the card. The card appears ONLY
  // when the entry is checked. There is no automatic fallback: with every box
  // unchecked a tap opens nothing (an unchecked control that behaves checked
  // reads as a bug — it was reported as one from a phone). Canvas click
  // coords are unreliable on the software-GL sandbox, so assert the predicate
  // and drive the card directly.
  await expect(page.locator("#toggle-pixel")).not.toBeChecked();
  expect(await page.evaluate(() => window.__earth.pixelInspectorEngaged())).toBe(false);
  await page.check("#toggle-pixel");
  expect(await page.evaluate(() => window.__earth.pixelInspectorEngaged())).toBe(true);
  // …and it lists as an active chip like any other layer
  await expect(page.locator("#active-layers .chip", { hasText: "Everything we know" })).toBeVisible();
  await page.uncheck("#toggle-pixel");
  // no colormapped layer active → STILL not engaged; the checkbox is the intent
  await page.uncheck('#layer-list input[data-id="sst"]');
  expect(await page.evaluate(() => window.__earth.pixelInspectorEngaged())).toBe(false);
  await page.check('#layer-list input[data-id="sst"]');
  expect(await page.evaluate(() => window.__earth.pixelInspectorEngaged())).toBe(false);

  // -- an ocean point in the North Atlantic
  await page.evaluate(() =>
    window.__earth.showPixelState(Cesium.Cartographic.fromDegrees(-30, 40)));
  const card = page.locator("#pixel-card");
  await expect(card).toBeVisible();
  await expect(card).toContainText("Pixel state");
  await expect(card).toContainText("40.00°N 30.00°W");
  // live weather + the future axis
  await expect(card).toContainText("Open-Meteo", { timeout: 60000 });
  await expect(card).toContainText("Air temperature");
  await expect(card.locator(".px-forecast .px-day")).toHaveCount(7);
  // satellite state at the app date; the annual mean is its own line (a
  // derived "vs annual mean" delta would mostly be the seasonal cycle)
  await expect(card).toContainText("Sea surface temperature");
  // The years moved out of the label and into the row's own stamp, read from
  // the baked file rather than typed into the UI (see the provenance test).
  await expect(card).toContainText("SST annual mean");
  await expect(card).toContainText("1991–2020");
  expect(await card.textContent()).not.toContain("vs 1991–2020");
  // Every value says WHEN it was observed, and the satellite heading no longer
  // claims one date for the whole block — because it never was one date. CERES
  // tiles stop at 2018-10, so that row must say 2018-10 while it sits beside
  // rows read today; under the old shared heading it read as current.
  const satHead = card.locator(".px-sec-title", { hasText: "Satellite fields" });
  await expect(satHead).toContainText("NASA GIBS");
  expect(await satHead.textContent()).not.toContain(
    await page.evaluate(() => window.__earth.state.date));
  const ceres = card.locator(".px-row", { hasText: "Energy balance" });
  await expect(ceres.locator(".px-when")).toContainText("2018-10");
  expect(await card.locator(".px-when").count()).toBeGreaterThan(8);

  // context: floats and monitoring sites exist in the North Atlantic
  await expect(card).toContainText("Argo floats");
  await expect(card).toContainText("Nearest monitoring site");
  // the role map is one click away
  await expect(card.locator('a[href="docs/PIXEL_STATE.md"]')).toHaveCount(1);

  // ocean bonus channels: waves (marine API) and CAMS air quality (global)
  await expect(card).toContainText("Waves");
  await expect(card).toContainText("Air quality");

  // -- a Swiss alpine point: regional normals + elevation + land channels
  await page.evaluate(() =>
    window.__earth.showPixelState(Cesium.Cartographic.fromDegrees(8.0, 46.5)));
  await expect(card).toContainText("Elevation", { timeout: 60000 });
  await expect(card).toContainText("MeteoSwiss");
  await expect(card).toContainText("E-OBS");
  await expect(card).toContainText("Soil (top cm)");
  // (no NDVI assertion: at ~3000 m this pixel is masked rock/snow, and the
  // row honestly omits itself — the layer itself is covered in the
  // state-vector test below)
  await expect(card).toContainText("PM2.5");
  await expect(card).toContainText("River discharge");
  // the decadal future axis: this pixel's own 2050 trajectory with model range
  await expect(card).toContainText("Projected change");
  await expect(card).toContainText("vs 1991–1995");
  await expect(card).toContainText("2045–2049");
  await expect(card).toContainText("models");

  // × closes it
  await page.click("#pixel-card .px-close");
  await expect(card).toBeHidden();
});

test("state-vector layers: end-clamping, 5-day snap, and a live toggle", async ({ page }) => {
  const t = await page.evaluate(() => {
    const E = window.__earth;
    const cfg = (id) => E.GIBS_LAYERS.find((l) => l.id === id);
    return {
      // archives that stop being served clamp to their last date...
      graceToday: E.gibsTime(cfg("grace"), E.state.date),
      ceresToday: E.gibsTime(cfg("ceres"), E.state.date),
      smToday: E.gibsTime(cfg("soilmoisture"), E.state.date),
      // ...but dates inside the archive pass through (with monthly snap)
      graceMid: E.gibsTime(cfg("grace"), "2010-06-15"),
      // 5-day product: floor to a valid epoch of the right anchor
      sshOld: E.gibsTime(cfg("ssh-anom"), "2000-01-04"),   // epoch1 1992-09-30
      sshNew: E.gibsTime(cfg("ssh-anom"), "2018-06-15"),   // epoch2 2017-10-29
      sshToday: E.gibsTime(cfg("ssh-anom"), E.state.date), // clamps then snaps
      // NDVI is current: today snaps to a recent month, not an endTime
      ndviToday: E.gibsTime(cfg("ndvi"), E.state.date),
    };
  });
  expect(t.graceToday).toBe("2022-07-01");
  expect(t.ceresToday).toBe("2018-10-01");
  expect(t.smToday).toBe("2025-09-01");
  expect(t.graceMid).toBe("2010-06-01");
  // 1992-09-30 + k*5d lands on 2000-01-02 for the 2000-01-04 request
  expect(t.sshOld).toBe("2000-01-02");
  // 2017-10-29 + k*5d: 2018-06-11 is the floor of 2018-06-15
  expect(t.sshNew).toBe("2018-06-11");
  // The last served epoch — MEASURED from the layer's GIBS time domain. It was
  // typed as 2019-01-17 for months, one 5-day step early, quietly hiding the
  // final frame of the archive. See "GIBS time domains" below.
  expect(t.sshToday).toBe("2019-01-22");
  expect(t.ndviToday >= "2026-01-01").toBe(true);
  // one of the five actually toggles on with our GIBS tiling scheme
  await page.check('#layer-list input[data-id="ndvi"]');
  const scheme = await page.evaluate(() =>
    window.__earth.state.layers["ndvi"].layer.imageryProvider.tilingScheme.constructor.name);
  expect(scheme).toBe("GIBSGeographicTilingScheme");
  // and carries a legend with its colormap
  await expect(page.locator("#legend-panel")).toContainText("Vegetation index");
});

test("argo 300 m anomaly layer + the card's ocean column", async ({ page }) => {
  test.setTimeout(150000);
  // the snapshot grid toggles on with a diverging legend and an honest toast
  await page.check('#layer-list input[data-id="argo-t300"]');
  await expect(page.locator("#legend-panel")).toContainText("Subsurface temp anomaly");
  const toast = page.locator("#toast-host .toast").first();
  await expect(toast).toBeVisible();
  await expect(toast).toContainText("recent-month snapshot");
  // grid probe reads a physical value somewhere in the subtropical gyre
  const probe = await page.evaluate(() =>
    window.__earth.probeValueAt(Cesium.Cartographic.fromDegrees(-40, 35)));
  expect(probe.units).toBe("°C");
  if (!probe.noData) expect(Math.abs(probe.value)).toBeLessThan(8);

  // column sampling: ocean cell has a full stratified profile, land has none
  const col = await page.evaluate(async () => {
    const oc = await (await fetch("data/ocean_column.json")).json();
    const E = window.__earth;
    return {
      pac: E.oceanColumnAt(oc, -170, 0),
      sahara: E.oceanColumnAt(oc, 10, 21),
      month: oc.month,
    };
  });
  expect(col.sahara).toBeNull();
  expect(col.pac.tNow[0]).toBeGreaterThan(20);            // warm tropical surface
  expect(col.pac.tNow[col.pac.tNow.length - 1]).toBeLessThan(6);  // cold abyss
  expect(col.month).toMatch(/^\d{4}-\d{2}$/);

  // the pixel card renders the profile for an ocean click...
  await page.evaluate(() =>
    window.__earth.showPixelState(Cesium.Cartographic.fromDegrees(-30, 40)));
  const card = page.locator("#pixel-card");
  await expect(card).toContainText("Ocean column 0–2000 m", { timeout: 60000 });
  await expect(card.locator("svg.px-profile")).toHaveCount(1);
  await expect(card).toContainText("Upper 700 m vs normal");
  await expect(card).toContainText("Surface salinity");
  // ...and not for a landlocked click (Bern)
  await page.evaluate(() =>
    window.__earth.showPixelState(Cesium.Cartographic.fromDegrees(7.45, 46.95)));
  await expect(card).toContainText("Elevation", { timeout: 60000 });
  await expect(card).not.toContainText("Ocean column");
});

test("GLORYS layers toggle; the card gets ocean circulation", async ({ page }) => {
  test.setTimeout(120000);
  await page.check('#layer-list input[data-id="currents"]');
  await page.check('#layer-list input[data-id="mld"]');
  await expect(page.locator("#legend-panel")).toContainText("Surface current speed");
  await expect(page.locator("#legend-panel")).toContainText("Mixed-layer depth");
  // probe reads the Gulf Stream in physical units
  const probe = await page.evaluate(() =>
    window.__earth.probeValueAt(Cesium.Cartographic.fromDegrees(-55, 58)));
  expect(probe.units).toBe("m");                      // topmost = MLD
  // card: an open-ocean click shows circulation with a compass direction
  await page.evaluate(() =>
    window.__earth.showPixelState(Cesium.Cartographic.fromDegrees(-74, 36)));
  const card = page.locator("#pixel-card");
  await expect(card).toContainText("Ocean circulation", { timeout: 60000 });
  await expect(card).toContainText("Surface current");
  await expect(card).toContainText("toward");
  await expect(card).toContainText("Mixed-layer depth");
});

test("GIBS time domains: parse the archive, snap into it, name the gap", async ({ page }) => {
  // Pure logic first, on hand-written domains that reproduce every shape GIBS
  // actually publishes — no network. The bug this closes: the app asked every
  // timed layer for "two days ago" and got a 404 whenever the archive lagged
  // (NDVI's newest monthly composite was 62 days old) or had an interior hole
  // (NDVI has no 2025-04). A blank globe with a legend and a probe reading
  // "no data" is the worst possible way to say "that date isn't published".
  const logic = await page.evaluate(() => {
    const E = window.__earth;
    const dom = (s) => E.parseGibsDomain(`<Domain>${s}</Domain>`);
    // NDVI's real shape: a long run, a hole, then a second run that lags today
    const ndvi = dom("2000-03-01/2025-03-01/P1M,2025-05-01/2026-06-01/P1M");
    // GRACE ships a malformed interval whose end precedes its start
    const grace = dom("2020-01-20/2020-01-10/P1M,2015-04-12/2015-06-12/P28D");
    const halfHour = dom("2026-08-01T00:00:00Z/2026-08-01T23:30:00Z/PT30M");
    const annual = dom("2023-01-01/2025-01-01/P1Y");
    return {
      periods: ["P1M", "P1Y", "P5D", "PT30M", "nonsense"].map((p) => E.parsePeriod(p)),
      ndviLen: ndvi.length,
      lag: E.snapToDomain(ndvi, "2026-08-01"),      // past the newest → the newest
      hole: E.snapToDomain(ndvi, "2025-04-15"),     // in the gap → newest before it
      exact: E.snapToDomain(ndvi, "2025-03-01"),    // served → unchanged
      early: E.snapToDomain(ndvi, "1999-01-01"),    // before the record → its start
      // one bad row must not cost the layer its whole domain
      graceOrdered: grace.map((i) => i.s),
      graceBad: E.snapToDomain(grace, "2020-01-25"),
      graceStep: E.snapToDomain(grace, "2015-05-01"),   // 28-day step, not a month
      subDaily: E.snapToDomain(halfHour, "2026-08-01T13:45:00Z"),
      annual: E.snapToDomain(annual, "2026-01-01"),
      garbage: E.parseGibsDomain("<html>nope</html>"),
    };
  });
  expect(logic.periods).toEqual([
    { months: 1, ms: 0 }, { months: 12, ms: 0 },
    { months: 0, ms: 5 * 864e5 }, { months: 0, ms: 30 * 60000 },
    { months: 0, ms: 864e5 },                       // unparseable → assume daily
  ]);
  expect(logic.ndviLen).toBe(2);
  expect(logic.lag).toBe("2026-06-01");
  expect(logic.hole).toBe("2025-03-01");
  expect(logic.exact).toBe("2025-03-01");
  expect(logic.early).toBe("2000-03-01");
  expect(logic.graceOrdered).toEqual(["2015-04-12", "2020-01-20"]);
  expect(logic.graceBad).toBe("2020-01-20");
  expect(logic.graceStep).toBe("2015-04-12");       // 2015-05-10 is the next step
  expect(logic.subDaily).toBe("2026-08-01T13:30:00Z");
  expect(logic.annual).toBe("2025-01-01");
  expect(logic.garbage).toBeNull();

  // Then the real thing: switch NDVI on, let its domain arrive, and check the
  // date we ask GIBS for is one the archive says it serves.
  const toasts = await recordToasts(page);         // see recordToasts: toasts expire
  await page.check('#layer-list input[data-id="ndvi"]');
  await expect
    .poll(() => page.evaluate(() => !!window.__earth.gibsDomains.get("ndvi")), { timeout: 30000 })
    .toBe(true);
  const real = await page.evaluate(() => {
    const E = window.__earth;
    const cfg = E.GIBS_LAYERS.find((l) => l.id === "ndvi");
    const dom = E.gibsDomains.get("ndvi");
    return {
      shown: E.gibsTime(cfg, E.state.date),
      asked: E.gibsTimeStatic(cfg, E.state.date),
      last: dom[dom.length - 1].e,
      served: dom.some((iv) => iv.s <= E.gibsTime(cfg, E.state.date) &&
                               E.gibsTime(cfg, E.state.date) <= iv.e),
      url: E.gibsDomainUrl(cfg),
    };
  });
  expect(real.url).toContain("/MODIS_Terra_L3_NDVI_Monthly/default/1km/all/all.xml");
  expect(real.served).toBe(true);                  // inside a published interval
  expect(real.shown <= real.asked).toBe(true);     // never asks for the future
  expect(real.shown).toBe(real.last);              // NDVI lags; it lands on the edge
  // and the layer paints at that month rather than sitting silently blank
  await expect(page.locator("#legend-panel")).toContainText("Vegetation index");
  // If the served month is behind the date we asked for, the user is told which
  // month is on screen — never pinned to an exact string, only to the fact that
  // the toast names the real month.
  if (real.shown !== real.asked) {
    await expect.poll(toasts).toContain(real.shown.slice(0, 7));
  }

  // A lagging archive and a hole in one are different facts and must read
  // differently — "not published yet" vs "there is a gap here". Both are driven
  // off a synthetic domain so the assertions don't depend on what NASA happens
  // to have published today; the months are computed from the app's own request.
  const months = await page.evaluate(() => {
    const E = window.__earth;
    const cfg = E.GIBS_LAYERS.find((l) => l.id === "ndvi");
    const asked = E.gibsTimeStatic(cfg, E.state.date, { clampEnd: false });
    const shift = (iso, n) => {
      const d = new Date(`${iso}T00:00:00Z`);
      d.setUTCMonth(d.getUTCMonth() + n);
      return d.toISOString().slice(0, 10);
    };
    return { asked, before: shift(asked, -2), after: shift(asked, 1), later: shift(asked, 5) };
  });
  const flip = (v) => page.evaluate((on) => {
    const el = document.querySelector('#layer-list input[data-id="ndvi"]');
    el.checked = on;
    el.dispatchEvent(new Event("change", { bubbles: true }));
  }, v);
  const setDomain = (xml) => page.evaluate((s) => {
    const E = window.__earth;
    E.gibsDomains.set("ndvi", E.parseGibsDomain(`<Domain>${s}</Domain>`));
  }, xml);
  const clearToasts = async () => {
    // Keep clicking closes until the host is empty: the domain-load callback
    // fires maybeArchiveToast asynchronously, so under a slow proxy a NEW
    // toast can land AFTER a single sweep of close-clicks and sit through the
    // assertion (seen once under CPU contention — resolved only at the 8s
    // auto-dismiss, after the 5s expect gave up).
    await expect
      .poll(() => page.evaluate(() => {
        const t = document.querySelectorAll("#toast-host .toast");
        for (const b of document.querySelectorAll("#toast-host .toast .toast-close")) b.click();
        return t.length;
      }), { timeout: 15000 })
      .toBe(0);
  };

  await flip(false);
  await clearToasts();
  await setDomain(`2000-03-01/${months.before}/P1M,${months.after}/${months.later}/P1M`);
  await flip(true);
  const hole = page.locator("#toast-host .toast").last();
  await expect(hole).toContainText("a gap in the archive");
  await expect(hole).toContainText(months.before.slice(0, 7));   // what you get instead
  await expect(hole).toContainText(months.asked.slice(0, 7));    // what you asked for

  await flip(false);
  await clearToasts();
  await setDomain(`2000-03-01/${months.before}/P1M`);            // lagging, no hole
  await flip(true);
  const lag = page.locator("#toast-host .toast").last();
  await expect(lag).toContainText("hasn't published");
  await expect(lag).toContainText("2 months behind");
  await expect(lag).toContainText(months.before.slice(0, 7));
});

test("GLORYS grids are month-keyed: the date's month picks the map", async ({ page }) => {
  // Month resolution semantics on a synthetic grid: floor within range,
  // clamp outside it — pure logic, no network.
  const picks = await page.evaluate(() => {
    const E = window.__earth;
    const g = { west: 0, south: 0, east: 2, north: 1, dlon: 1, dlat: 1, nx: 2, ny: 1,
                months: { "2025-11": [11, 11], "2026-01": [1, 1], "2026-05": [5, 5] } };
    const at = (d) => { E.state.date = d; return [E.resolveGridMonth(g), E.sampleGrid(g, 0.5, 0.5)]; };
    const out = {
      exact: at("2026-01-15"),      // inside a baked month → that month
      floor: at("2026-03-10"),      // between baked months → newest earlier one
      before: at("2024-06-01"),     // before the range → clamps to earliest
      after: at("2027-01-01"),      // after the range → clamps to latest
    };
    E.state.date = new Date().toISOString().slice(0, 10);
    return out;
  });
  expect(picks.exact).toEqual(["2026-01", 1]);
  expect(picks.floor).toEqual(["2026-01", 1]);
  expect(picks.before).toEqual(["2025-11", 11]);
  expect(picks.after).toEqual(["2026-05", 5]);

  // The real baked file declares months, and enabling the layer says which
  // month is showing (monthly-toast, not the generic dateless one).
  await page.check('#layer-list input[data-id="currents"]');
  const toast = page.locator(".toast", { hasText: "monthly-mean" });
  await expect(toast).toContainText("month does");
  const info = await page.evaluate(async () => {
    const E = window.__earth;
    const cfg = E.GIBS_LAYERS.find((l) => l.id === "currents");
    const g = await E.loadGrid(cfg);
    return { months: E.gridMonths(g), shown: E.resolveGridMonth(g),
             dateless: E.datelessToast("currents") };
  });
  expect(info.months.length).toBeGreaterThanOrEqual(300); // full 1993→ archive
  expect(info.dateless).toBeNull();                       // no false "snapshot" claim
  await expect(toast).toContainText(info.shown);          // the toast names the month
  // probe sampling agrees with the resolved month's field (Gulf Stream is fast)
  const probe = await page.evaluate(() =>
    window.__earth.probeValueAt(Cesium.Cartographic.fromDegrees(-74, 36)));
  expect(probe.value).toBeGreaterThan(0.5);
  // history lazy-loads: an old month lives in a per-year file that is fetched
  // on demand and sampled through the same path
  const old = await page.evaluate(async () => {
    const E = window.__earth;
    E.state.date = "2005-06-15";
    const cfg = E.GIBS_LAYERS.find((l) => l.id === "currents");
    const g = await E.loadGridMonth(cfg);
    const out = { m: E.resolveGridMonth(g), v: E.sampleGrid(g, -74, 36) };
    E.state.date = new Date().toISOString().slice(0, 10);
    return out;
  });
  expect(old.m).toBe("2005-06");
  expect(old.v).toBeGreaterThan(0.25);                    // the Stream was there in 2005 too
});

test("temperature scene probe: SST under LST, and kelvin reads as °C", async ({ page }) => {
  test.setTimeout(150000);
  // the temperature scene stacks LST (land) over SST (ocean)
  await page.evaluate(() => window.__earth.SCENES && document.querySelector('[data-scene="temperature"]').click());
  await page.waitForFunction(() =>
    window.__earth.state.layers.sst?.layer && window.__earth.state.layers.lst?.layer);
  // ocean point: LST (top) is transparent there — the probe must fall
  // through to SST instead of reporting nothing (user-reported bug)
  const ocean = await page.evaluate(() =>
    window.__earth.probeValueAt(Cesium.Cartographic.fromDegrees(-40, 30)));
  expect(ocean.noData).toBeFalsy();
  expect(ocean.title).toContain("Sea surface temperature");
  expect(ocean.value).toBeGreaterThan(-2);
  expect(ocean.value).toBeLessThan(35);
  // land point: LST answers, converted from the colormap's kelvin to °C
  const land = await page.evaluate(() =>
    window.__earth.probeValueAt(Cesium.Cartographic.fromDegrees(10, 21)));   // Sahara
  if (!land.noData) {                       // clear-sky product: gaps possible
    expect(land.title).toContain("Land surface temperature");
    expect(land.units).toBe("°C");
    expect(land.value).toBeGreaterThan(-40);
    expect(land.value).toBeLessThan(75);    // a kelvin leak would read ~300+
  }
  // pure unit check, gap-proof: kelvinToC converts absolutes, spares deltas
  const conv = await page.evaluate(() => [
    window.__earth.kelvinToC({ value: 300, units: "K" }),
    window.__earth.kelvinToC({ value: 3, units: "K", delta: true }),
  ]);
  expect(conv[0].value).toBeCloseTo(26.85, 2);
  expect(conv[0].units).toBe("°C");
  expect(conv[1].value).toBe(3);            // Δ kelvin == Δ °C, no offset
});

test("GFS forecast layers open the date selector to the future", async ({ page }) => {
  const today = new Date().toISOString().slice(0, 10);
  await page.check('#layer-list input[data-id="gfs-temp"]');
  const toast = page.locator(".toast", { hasText: "forecast" });
  await expect(toast).toContainText("future");
  await expect(toast).toContainText("GFS run");
  // once the grid loads, the max selectable date is the last forecast day
  await page.waitForFunction((d) =>
    document.getElementById("layer-date").max > d, today);
  const info = await page.evaluate(async () => {
    const E = window.__earth;
    const cfg = E.GIBS_LAYERS.find((l) => l.id === "gfs-temp");
    const g = await E.loadGridMonth(cfg);
    return { latest: g.latest, keyLen: g.keyLen,
             max: document.getElementById("layer-date").max };
  });
  expect(info.keyLen).toBe(10);
  expect(info.max).toBe(info.latest);
  expect(info.latest > today).toBe(true);
  // jump to the last forecast day: the frame resolves to that exact day and
  // carries physical values; observation layers clamp back to real time
  await page.fill("#layer-date", info.latest);
  await page.dispatchEvent("#layer-date", "change");
  const res = await page.evaluate(async () => {
    const E = window.__earth;
    const cfg = E.GIBS_LAYERS.find((l) => l.id === "gfs-temp");
    const g = await E.loadGridMonth(cfg);
    const viirs = E.GIBS_LAYERS.find((l) => l.id === "viirs-truecolor");
    return { m: E.resolveGridMonth(g), tropics: E.sampleGrid(g, -140, 0),
             viirsDate: E.gibsTime(viirs, E.state.date) };
  });
  expect(res.m).toBe(info.latest);
  expect(res.tropics).toBeGreaterThan(15);                // warm Pacific, any season
  expect(res.viirsDate <= today).toBe(true);              // GIBS never asked for tomorrow
  // switching the forecast off pulls the date back into the observed record
  await page.uncheck('#layer-list input[data-id="gfs-temp"]');
  await page.waitForFunction((d) => {
    const el = document.getElementById("layer-date");
    return el.max <= d && el.value <= d;
  }, today);
});

test("layer opacity: labeled slider with a ½ toggle for overlaying fields", async ({ page }) => {
  // the motivating use: currents below, SST at 50% on top — do they line up?
  await page.check('#layer-list input[data-id="currents"]');
  await page.check('#layer-list input[data-id="sst"]');
  const row = page.locator('[data-alpharow="sst"]');
  await expect(row).toBeVisible();
  await expect(row.locator(".alpha-val")).toHaveText("100%");
  await row.locator(".alpha-half").click();
  await expect(row.locator(".alpha-val")).toHaveText("50%");
  expect(await page.evaluate(() => window.__earth.state.layers.sst.layer.alpha)).toBeCloseTo(0.5, 5);
  // the currents layer underneath keeps its own opacity
  expect(await page.evaluate(() => window.__earth.state.layers.currents.layer.alpha)).toBe(1);
  await row.locator(".alpha-half").click();               // toggles back
  await expect(row.locator(".alpha-val")).toHaveText("100%");
  // the slider drives the same path, any value
  await row.locator('input[data-alpha="sst"]').fill("30");
  await expect(row.locator(".alpha-val")).toHaveText("30%");
  expect(await page.evaluate(() => window.__earth.state.layers.sst.layer.alpha)).toBeCloseTo(0.3, 5);
});

test("archive-end comparisons explain their emptiness instead of hiding it", async ({ page }) => {
  // CERES tiles end 2018-10; with today's date, "now" and "2 years ago" both
  // clamp there → a zero-by-construction comparison. The hint must say so.
  await page.check('#layer-list input[data-id="ceres"]');
  await page.selectOption("#compare-select", "2");
  const hint = page.locator("#delta-hint");
  await expect(hint).toBeVisible();
  await expect(hint).toContainText("archive ends 2018-10");
  await expect(hint).toContainText("empty by construction");
  // moving the date inside the archive clears the warning and compares for real
  await page.evaluate(() => {
    const d = document.getElementById("layer-date");
    d.value = "2018-09-15";
    d.dispatchEvent(new Event("change"));
  });
  await expect(hint).not.toContainText("empty by construction");
  const t = await page.evaluate(() => {
    const E = window.__earth;
    const cfg = E.GIBS_LAYERS.find((l) => l.id === "ceres");
    return { now: E.gibsTime(cfg, E.state.date), past: E.gibsTime(cfg, E.compareDate()) };
  });
  expect(t.now).toBe("2018-09-01");
  expect(t.past).toBe("2016-09-01");   // genuinely different months → real delta
});

test("Energy tab shows Earth's energy imbalance with plausible numbers", async ({ page }) => {
  await page.click("#tab-energy");
  await expect(page.locator("#eei-rate .stat-value")).not.toHaveText("–");
  const stats = await page.evaluate(() => ({
    rate: parseFloat(document.querySelector("#eei-rate .stat-value").textContent),
    total: parseFloat(document.querySelector("#eei-total .stat-value").textContent),
    zj: parseFloat(document.querySelector("#eei-zj .stat-value").textContent),
  }));
  expect(stats.rate).toBeGreaterThan(0.4);
  expect(stats.rate).toBeLessThan(1.3);
  expect(stats.total).toBeGreaterThan(stats.rate);     // /0.9 → always larger
  expect(stats.zj).toBeGreaterThan(150);
  // the chart drew both OHC curves
  const px = await page.evaluate(() => {
    const c = document.getElementById("eei-chart");
    const d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
    let n = 0;
    for (let i = 3; i < d.length; i += 4) if (d[i] > 0) n++;
    return n;
  });
  expect(px).toBeGreaterThan(1000);
  // the second chart draws the imbalance ITSELF over time (the slope)
  const rpx = await page.evaluate(() => {
    const c = document.getElementById("eei-rate-chart");
    const d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
    let n = 0;
    for (let i = 3; i < d.length; i += 4) if (d[i] > 0) n++;
    return n;
  });
  expect(rpx).toBeGreaterThan(1000);
  await expect(page.locator("#panel-energy")).toContainText("90 %");
  await expect(page.locator("#panel-energy")).toContainText("Y axis: accumulated heat");
  await expect(page.locator("#panel-energy")).toContainText("watts per m²");
  // ENSO/volcano annotations: legend names them, bands paint both hues
  // (amber = warm-channel-dominant, teal = blue-channel-dominant)
  await expect(page.locator("#eei-rate-legend")).toContainText("El Niño");
  await expect(page.locator("#eei-rate-legend")).toContainText("Pinatubo");
  const tint = await page.evaluate(() => {
    const c = document.getElementById("eei-rate-chart");
    const d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
    let red = 0, blue = 0;
    for (let i = 0; i < d.length; i += 4) {
      if (d[i + 3] === 0) continue;
      if (d[i] > d[i + 2] + 8) red++;
      else if (d[i + 2] > d[i] + 8) blue++;
    }
    return { red, blue };
  });
  expect(tint.red).toBeGreaterThan(500);
  expect(tint.blue).toBeGreaterThan(500);
  // the intro quotes the SAME numbers as the tiles, with the data's window
  const introTotal = await page.textContent("#eei-intro-total");
  const tileTotal = await page.textContent("#eei-total .stat-value");
  expect(introTotal).toBe(tileTotal);
  expect(await page.textContent("#eei-intro-window")).toMatch(/^\d{4}–\d{4}$/);
  await expect(page.locator("#panel-energy")).toContainText("needs its date attached");
  // forcing vs imbalance: the IPCC's ~2.8 W/m² (radiative forcing) is
  // reconciled in the text, not left to look like a contradiction
  await expect(page.locator("#panel-energy")).toContainText("radiative forcing");
  await expect(page.locator("#panel-energy")).toContainText("2.8");
  await expect(page.locator("#panel-energy")).toContainText("not yet answered");
  // the push curves are drawn and named, and the counterfactuals documented
  await expect(page.locator("#eei-rate-legend")).toContainText("human push (total ERF)");
  await expect(page.locator("#eei-rate-legend")).toContainText("natural push");
  await expect(page.locator("#panel-energy")).toContainText("vanished tomorrow");
  await expect(page.locator("#panel-energy")).toContainText("never industrialized");
  await expect(page.locator("#panel-energy")).toContainText("not");
  const erfDrawn = await page.evaluate(async () => {
    const d = await fetch("data/eei.json?v=3").then((r) => r.json());
    // taller canvas is the visible signature that the ERF curves rendered
    const h = document.getElementById("eei-rate-chart").style.height;
    return { has: Array.isArray(d.erf_years) && d.erf_years.length > 50, h };
  });
  expect(erfDrawn.has).toBe(true);
  expect(parseInt(erfDrawn.h)).toBeGreaterThan(180);
  // Hiroshima equivalence: computed from the live EEI, in a plausible band
  const bombs = parseFloat(await page.textContent("#eei-bombs"));
  expect(bombs).toBeGreaterThan(3);
  expect(bombs).toBeLessThan(15);
  await expect(page.locator("#panel-energy")).toContainText("nothing explodes");
  // slab decomposition: 0-700 nests inside 0-2000, difference = 700-2000 m
  await expect(page.locator("#eei-legend")).toContainText("700–2000 m slab");
  const slab = await page.evaluate(() => {
    const E = window.__earth; // eeiData isn't exported; recompute from the file
    return fetch("data/eei.json?v=3").then((r) => r.json()).then((d) => {
      const i = d.y2000.length - 1;
      const deep = d.ohc2000[i] - d.ohc700[d.y2000[i] - d.y700[0]];
      return { deep, total: d.ohc2000[i], upper: d.ohc700[d.y2000[i] - d.y700[0]] };
    });
  });
  expect(slab.deep).toBeGreaterThan(0);              // the deep layer HAS gained heat
  expect(slab.deep).toBeLessThan(slab.total);        // …and is a proper subset of it
  // smoothing: slider is source of truth, presets snap it; 1y default
  await expect(page.locator('#eei-smooth button[data-n="1"]')).toHaveClass(/active/);
  await page.locator("#eei-smooth-slider").fill("7");
  await expect(page.locator("#eei-smooth-value")).toHaveText("7 yr");
  await expect(page.locator("#eei-smooth button.active")).toHaveCount(0);  // off-preset value
  const hash = () => page.evaluate(() => {
    const d = document.getElementById("eei-rate-chart").getContext("2d")
      .getImageData(0, 0, 200, 200).data;
    let h = 0;
    for (let i = 0; i < d.length; i += 97) h = (h * 31 + d[i]) >>> 0;
    return h;
  });
  const ledgerHash = () => page.evaluate(() => {
    const d = document.getElementById("eei-chart").getContext("2d")
      .getImageData(0, 0, 200, 200).data;
    let h = 0;
    for (let i = 0; i < d.length; i += 97) h = (h * 31 + d[i]) >>> 0;
    return h;
  });
  const before = await hash();
  const ledgerBefore = await ledgerHash();
  await page.click('#eei-smooth button[data-n="10"]');
  await expect(page.locator('#eei-smooth button[data-n="10"]')).toHaveClass(/active/);
  expect(await page.inputValue("#eei-smooth-slider")).toBe("10");  // button snapped the slider
  expect(await hash()).not.toBe(before);        // the rate chart follows the window...
  expect(await ledgerHash()).toBe(ledgerBefore); // ...the raw ledger does not
});

test("sidebar is resizable by dragging, persists, and resets on double-click", async ({ page }) => {
  const width = () => page.evaluate(() => document.getElementById("sidebar").offsetWidth);
  expect(await width()).toBe(380);                     // new, wider default
  // drag the handle 120px to the right
  const h = await page.locator("#sidebar-resize").boundingBox();
  await page.mouse.move(h.x + 3, h.y + 300);
  await page.mouse.down();
  await page.mouse.move(500, h.y + 300, { steps: 5 });
  await page.mouse.up();
  expect(await width()).toBe(500);
  // persisted: a reload keeps it
  expect(await page.evaluate(() => localStorage.getItem("sidebarW"))).toBe("500");
  await page.reload();
  await page.waitForFunction(() => window.__earth?.viewer, null, { timeout: 30000 });
  expect(await width()).toBe(500);
  // clamped: absurd drags stop at the max (60% of the window)
  const h2 = await page.locator("#sidebar-resize").boundingBox();
  await page.mouse.move(h2.x + 3, h2.y + 300);
  await page.mouse.down();
  await page.mouse.move(1200, h2.y + 300, { steps: 5 });
  await page.mouse.up();
  const w = await width();
  expect(w).toBeLessThanOrEqual(1280 - 240);   // structural max: keep some globe
  expect(w).toBeGreaterThan(680);              // the old aesthetic cap is gone
  // double-click resets to the default and clears the preference
  await page.dblclick("#sidebar-resize");
  expect(await width()).toBe(380);
  expect(await page.evaluate(() => localStorage.getItem("sidebarW"))).toBeNull();
  // the globe container follows the variable
  const left = await page.evaluate(() =>
    document.getElementById("cesiumContainer").getBoundingClientRect().left);
  expect(Math.round(left)).toBe(380);
});

test("tagline scenes: one honest layer each, always replacing the last", async ({ page }) => {
  test.setTimeout(180000);
  const chips = page.locator("#active-layers .chip:not(.chip-clear)");
  const labels = () => page.locator("#active-layers .chip-label").allTextContents();
  const has = async (frag) => (await labels()).some((s) => s.includes(frag));
  const toasts = await recordToasts(page);   // see recordToasts: toasts expire

  // "sea ice": exactly ONE chip — no glacier inventory smuggled in
  await page.click('.tag-link[data-scene="seaice"]');
  await expect(chips).toHaveCount(1);
  expect(await has("Sea ice")).toBe(true);
  expect(await has("Sea surface temperature")).toBe(false);   // default swapped out
  expect(await has("Glaciers")).toBe(false);
  // the clamp toast fires on enable — asserted from the recorded log, because
  // the chip checks above can outlast the toast on a busy page
  await expect.poll(toasts).toContain("archive ends 2025-09");
  // the lesson of the blank-globe bug: a chip is not data. Assert that the
  // EFFECTIVE date (endTime-clamped) actually serves a tile with ice pixels,
  // that the clamp toast explains the older date, and that the camera flew
  // to the Arctic where the data is visible.
  const iceInfo = await page.evaluate(async () => {
    const E = window.__earth;
    const cfg = E.GIBS_LAYERS.find((l) => l.id === "seaice");
    const eff = E.gibsTime(cfg, E.state.date);
    // north-polar tile at level 2 (rows start at the north): x=2,y=0 covers the Arctic
    const url = `https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/${cfg.layer}` +
      `/default/${eff}/${cfg.tms}/2/0/2.png`;
    const img = await createImageBitmap(await (await fetch(url)).blob());
    const c = document.createElement("canvas");
    c.width = c.height = 512;
    const x = c.getContext("2d");
    x.drawImage(img, 0, 0);
    const d = x.getImageData(0, 0, 512, 512).data;
    let px = 0;
    for (let i = 3; i < d.length; i += 4) if (d[i] > 0) px++;
    const cam = Cesium.Cartographic.fromCartesian(E.viewer.camera.position);
    return { eff, px, camLat: Cesium.Math.toDegrees(cam.latitude) };
  });
  expect(iceInfo.eff).toBe("2025-09-01");            // clamped to the last served date
  expect(iceInfo.px).toBeGreaterThan(5000);          // and that date has real ice pixels
  await expect.poll(() => page.evaluate(() => {
    const c = Cesium.Cartographic.fromCartesian(window.__earth.viewer.camera.position);
    return Cesium.Math.toDegrees(c.latitude);
  }), { timeout: 15000 }).toBeGreaterThan(55);       // flew to the Arctic

  // "floats": the Argo fleet alone
  await page.click('.tag-link[data-scene="floats"]');
  await expect(chips).toHaveCount(1);
  expect(await has("Argo floats")).toBe(true);
  expect(await has("Sea ice")).toBe(false);
  await expect
    .poll(() => page.evaluate(() => window.__earth.pointLayers.argo?.collection.length ?? 0))
    .toBeGreaterThan(2000);

  // "vegetation": NDVI alone (no GBIF stacked underneath)
  await page.click('.tag-link[data-scene="vegetation"]');
  await expect(chips).toHaveCount(1);
  expect(await has("Vegetation")).toBe(true);
  expect(await has("Biodiversity")).toBe(false);

  // "emissions": the facilities alone (no AOD underneath)
  await page.click('.tag-link[data-scene="emissions"]');
  await expect(chips).toHaveCount(1);
  expect(await has("Facility emissions")).toBe(true);

  // "surface temperature": the ONE sanctioned two-layer scene — SST (ocean)
  // and LST (land) are spatially disjoint, composing one temperature field
  await page.click('.tag-link[data-scene="temperature"]');
  await expect(chips).toHaveCount(2);
  expect(await has("Sea surface temperature")).toBe(true);
  expect(await has("Land surface temperature")).toBe(true);
  expect(await has("Facility emissions")).toBe(false);

  // the no-stacking rule, enforced: single layers except the disjoint pair
  const scenes = await page.evaluate(() => window.__earth.SCENES ?? {});
  for (const [k, ids] of Object.entries(scenes)) {
    if (k === "temperature") expect(ids).toEqual(["sst", "lst"]);
    else expect(ids.length, `${k} single-layer`).toBe(1);
  }

  // "inspect any point": arms the inspector and says what it will show
  await page.click('.tag-link[data-scene="inspect"]');
  await expect(page.locator("#toggle-pixel")).toBeChecked();
  expect(await page.evaluate(() => window.__earth.pixelInspectorEngaged())).toBe(true);
  await expect.poll(toasts).toContain("2045–49");
});
test("Climate TRACE is year-aware: the date's year picks the inventory", async ({ page }) => {
  // start on a known in-range year
  await page.evaluate(() => {
    const d = document.getElementById("layer-date");
    d.value = "2024-07-15"; d.dispatchEvent(new Event("change"));
  });
  await page.check("#toggle-climatetrace");
  await expect
    .poll(() => page.evaluate(() => window.__earth.pointLayers.climatetrace?.collection.length ?? 0))
    .toBe(1000);
  // the enable toast explains the yearly semantics, not "date doesn't apply"
  const toast = page.locator("#toast-host .toast").last();
  await expect(toast).toContainText("yearly");
  await expect(toast).toContainText("2024");
  await expect(page.locator("#meta-climatetrace")).toContainText("2024 inventory");

  // a top-emitter marker's popup carries the shown year
  const before = await page.evaluate(() => window.__earth.pointLayers.climatetrace.__json.assets_by_year["2024"][0][3]);
  // step the year back two: the layer rebuilds for 2022
  await page.click('#date-steps button[data-step="-1y"]');
  await page.click('#date-steps button[data-step="-1y"]');
  await expect(page.locator("#meta-climatetrace")).toContainText("2022 inventory");
  const loadedYear = await page.evaluate(() => window.__earth.state.date.slice(0, 4));
  expect(loadedYear).toBe("2022");

  // stepping to a year outside the baked range clamps to the nearest edge
  await page.evaluate(() => {
    const d = document.getElementById("layer-date");
    d.value = "2010-03-01"; d.dispatchEvent(new Event("change"));
  });
  await expect(page.locator("#meta-climatetrace")).toContainText("2021 inventory"); // clamped up
});

test("OPERA disturbance layers are classifications: swatch legend, class-label probe", async ({ page }) => {
  // The classification parser on the real GIBS colormap. Continuous colormaps
  // carry value="[lo,hi)"; these carry sourceValue + a <Legend
  // type="classification">, so the continuous parser matches nothing and a
  // separate one is required — that's the whole reason `classmap` exists.
  const cm = await page.evaluate(async () => {
    const url = "https://gibs.earthdata.nasa.gov/colormaps/v1.3/OPERA_Vegetation_Disturbance_Status.xml";
    const xml = await (await fetch(url)).text();
    const cls = window.__earth.parseClassEntries(xml);
    const cont = window.__earth.parseColormapEntries(xml);
    return {
      n: cls?.classes.length ?? 0,
      labels: (cls?.classes ?? []).map((c) => c.label),
      rgbOk: (cls?.classes ?? []).every((c) => c.rgb.length === 3 && c.rgb.every((v) => v >= 0 && v <= 255)),
      contEntries: cont?.entries?.length ?? 0,
    };
  });
  expect(cm.n).toBeGreaterThan(2);
  expect(cm.rgbOk).toBe(true);
  expect(cm.labels.join(" | ")).toMatch(/confirmed/i);
  expect(cm.labels.some((l) => /^no data$/i.test(l))).toBe(false);  // transparent fill is not a class
  expect(cm.contEntries).toBe(0);                                    // continuous parser is blind here

  // rgb → label lookup, keyed on the exact packed colour
  const lut = await page.evaluate(async () => {
    const m = await window.__earth.getClassLut(
      "https://gibs.earthdata.nasa.gov/colormaps/v1.3/OPERA_Vegetation_Disturbance_Status.xml");
    return { size: m?.size ?? 0, values: [...(m?.values() ?? [])] };
  });
  expect(lut.size).toBe(cm.n);
  expect(lut.values.sort()).toEqual([...cm.labels].sort());

  // enable the layer → labelled swatches, NOT a gradient bar (the classes have
  // no ordering a bar could imply), plus the plain-language class note
  await page.evaluate(() => {
    const el = document.querySelector('#layer-list input[data-id="dist-alert"]');
    el.checked = true; el.dispatchEvent(new Event("change", { bubbles: true }));
  });
  const item = page.locator("#legend-panel .legend-item", { hasText: "disturbance" });
  await expect(item.locator(".legend-class")).toHaveCount(cm.n);
  await expect(item.locator(".legend-class .legend-swatch").first()).toBeVisible();
  await expect(item.locator("canvas.legend-bar")).toHaveCount(0);
  await expect(item.locator(".legend-note")).toContainText("confirmed");

  // the probe answers with a class NAME or no-data — never a number, and never
  // a delta/aggregate variant (class codes don't average or subtract)
  const probe = await page.evaluate(async () => {
    const e = window.__earth.colormapLayersTopDown().find((l) => l.cfg.id === "dist-alert");
    const at = (lon, lat) =>
      window.__earth.probeEntryValue(e, Cesium.Cartographic.fromDegrees(lon, lat));
    return { amazon: await at(-60, -9), ocean: await at(-30, 5) };
  });
  for (const r of [probe.amazon, probe.ocean]) {
    expect(r === null || r.noData === true || typeof r.label === "string").toBe(true);
    expect(r?.value).toBeUndefined();
    expect(r?.delta).toBeUndefined();
  }
  expect(probe.ocean.noData).toBe(true);   // open ocean has no vegetation to lose

  // posture matrix: classification rasters take neither flag
  const flags = await page.evaluate(() =>
    window.__earth.GIBS_LAYERS.filter((l) => l.classmap)
      .map((l) => [l.id, l.aggregable ?? null, l.deltaRange ?? null]));
  expect(flags.length).toBe(2);
  for (const [, agg, dr] of flags) { expect(agg).toBeNull(); expect(dr).toBeNull(); }
});

test("DIST-ANN is annual: the date's year picks the map, day and month don't", async ({ page }) => {
  const snap = await page.evaluate(() => {
    const cfg = window.__earth.GIBS_LAYERS.find((l) => l.id === "dist-ann");
    const t = (d) => window.__earth.gibsTime(cfg, d);
    return {
      mid: t("2024-07-15"), other: t("2024-11-02"),   // same year, different day/month
      next: t("2023-06-01"),
      early: t("2019-03-01"),                          // before the archive → floored
      late: t("2030-01-01"),                           // after → clamped to endTime's year
      start: cfg.start, endTime: cfg.endTime,
    };
  });
  expect(snap.mid).toBe("2024-01-01");
  expect(snap.other).toBe(snap.mid);                  // day and month are ignored
  expect(snap.next).toBe("2023-01-01");
  expect(snap.early).toBe("2023-01-01");
  expect(snap.late).toBe("2025-01-01");

  // enabling it says so: the year matters, the day and month don't. This is the
  // Climate TRACE trap one rung coarser — without the toast the steppers look broken.
  await page.evaluate(() => {
    const d = document.getElementById("layer-date");
    d.value = "2024-07-15"; d.dispatchEvent(new Event("change"));
    const el = document.querySelector('#layer-list input[data-id="dist-ann"]');
    el.checked = true; el.dispatchEvent(new Event("change", { bubbles: true }));
  });
  const toast = page.locator("#toast-host .toast").last();
  await expect(toast).toContainText("annual");
  await expect(toast).toContainText("2024");
  await expect(toast).toContainText("year");
});

test("drivers is a categorical grid: swatch legend, named driver, dateless", async ({ page }) => {
  // The palette lives in the baked FILE, not the layer config — so the legend,
  // the paint and the probe all have to agree with what the producer shipped.
  const g = await page.evaluate(async () => {
    const cfg = window.__earth.GIBS_LAYERS.find((l) => l.id === "drivers");
    const g = await window.__earth.loadGrid(cfg);
    return { classes: g.classes, nx: g.nx, ny: g.ny, len: g.values.length,
             sampleNulls: g.values.slice(0, 200).every((v) => v == null) };
  });
  expect(g.classes.length).toBe(7);
  expect(g.len).toBe(g.nx * g.ny);      // `packed` was expanded by unpackGrid…
  expect(g.sampleNulls).toBe(true);     // …and "." really became null, not 0

  await page.evaluate(() => {
    const el = document.querySelector('#layer-list input[data-id="drivers"]');
    el.checked = true; el.dispatchEvent(new Event("change", { bubbles: true }));
  });

  // Same swatch legend as the classification RASTERS — one shape for "the value
  // is a category" — and emphatically not a gradient bar.
  const item = page.locator("#legend-panel .legend-item", { hasText: "Drivers of forest loss" });
  await expect(item.locator(".legend-class")).toHaveCount(7);
  await expect(item.locator(".legend-class .legend-swatch").first()).toBeVisible();
  await expect(item.locator("canvas.legend-bar")).toHaveCount(0);
  await expect(item.locator(".legend-class")).toContainText([/agricult/i], { useInnerText: true });

  // Untimed and not a climatology: one attribution over the whole record, so
  // the toast has to say the date selector is inert here.
  const toast = page.locator("#toast-host .toast").last();
  await expect(toast).toContainText("2001");
  await expect(toast).toContainText("doesn't change it");

  // The probe answers with the driver's NAME. A bare "1" would tell a reader
  // nothing, and there is no number here to format.
  const probe = await page.evaluate(async () => {
    const e = window.__earth.colormapLayersTopDown().find((l) => l.cfg.id === "drivers");
    const at = (lon, lat) =>
      window.__earth.probeEntryValue(e, Cesium.Cartographic.fromDegrees(lon, lat));
    return { arc: await at(-55, -8), ocean: await at(-30, 5) };   // Amazon arc / open Atlantic
  });
  expect(typeof probe.arc.label).toBe("string");
  expect(probe.arc.value).toBeUndefined();
  expect(probe.ocean.noData).toBe(true);

  // posture matrix: categorical grids average and subtract no better than
  // categorical rasters do
  const cfg = await page.evaluate(() => {
    const c = window.__earth.GIBS_LAYERS.find((l) => l.id === "drivers");
    return { agg: c.aggregable ?? null, dr: c.deltaRange ?? null, cg: c.classGrid };
  });
  expect(cfg.cg).toBe(true);
  expect(cfg.agg).toBeNull();
  expect(cfg.dr).toBeNull();
});

test("the AMOC eval mask shows where the model computes, and says it is not a measurement", async ({ page }) => {
  // The one layer here that draws a MODEL rather than the world, so the thing
  // to protect is that a reader can tell those apart: three named roles, a
  // probe that answers with the role, and a toast that says the date does
  // nothing. Chris asked for it directly — "see which pixels will all be
  // rolled forward in the amoc eval" — and the answer is only useful if the
  // scored corridor is visibly a SUBSET of the rolled window.
  const g = await page.evaluate(async () => {
    const cfg = window.__earth.GIBS_LAYERS.find((l) => l.id === "amoc-eval");
    const g = await window.__earth.loadGrid(cfg);
    return { classes: g.classes, counts: g.counts, nx: g.nx, ny: g.ny,
             len: g.values.length, doc: cfg.doc };
  });
  expect(g.classes.map((c) => c.code)).toEqual([1, 2, 3]);
  expect(g.len).toBe(g.nx * g.ny);                 // packed → values
  expect(g.counts.corridor).toBeLessThan(g.counts.rolled);   // scored ⊂ rolled
  expect(g.counts.section).toBeLessThan(g.counts.corridor);
  // its "documentation" is the experiment's own plan, since there is no
  // third-party dataset behind it
  expect(g.doc).toMatch(/E022_spatial_coupling\.md$/);

  await page.evaluate(() => {
    const el = document.querySelector('#layer-list input[data-id="amoc-eval"]');
    el.checked = true; el.dispatchEvent(new Event("change", { bubbles: true }));
  });

  const item = page.locator("#legend-panel .legend-item", { hasText: "pixels rolled forward" });
  await expect(item.locator(".legend-class")).toHaveCount(3);
  await expect(item.locator("canvas.legend-bar")).toHaveCount(0);
  await expect(item.locator(".legend-class")).toContainText([/rolled/i], { useInnerText: true });

  // Dateless, but for its OWN reason: this is an experiment's fixed geometry,
  // not the drivers map's 25-year attribution. A shared sentence would be
  // false here, which is why the config carries its own note.
  const toast = page.locator("#toast-host .toast").last();
  await expect(toast).toContainText("geometry of an experiment");
  await expect(toast).toContainText("doesn't change it");

  // The probe answers with the ROLE. Cape Hatteras is scored corridor, the
  // RAPID row is section, and the Pacific side of the window has no state at
  // all — "no data" there is the honest answer, not a gap in a dataset.
  const probe = await page.evaluate(async () => {
    const e = window.__earth.colormapLayersTopDown().find((l) => l.cfg.id === "amoc-eval");
    const at = (lon, lat) =>
      window.__earth.probeEntryValue(e, Cesium.Cartographic.fromDegrees(lon, lat));
    return { gs: await at(-73, 36), sec: await at(-70, 26.5),
             deep: await at(-30, 40), out: await at(-140, 40) };
  });
  expect(probe.gs.label).toMatch(/corridor/i);
  expect(probe.sec.label).toMatch(/RAPID/i);
  expect(probe.deep.label).toMatch(/rolled/i);
  expect(probe.gs.value).toBeUndefined();          // a role is not a number
  expect(probe.out.noData).toBe(true);

  // posture: categorical AND untimed, so neither averaging nor differencing
  const cfg = await page.evaluate(() => {
    const c = window.__earth.GIBS_LAYERS.find((l) => l.id === "amoc-eval");
    return { agg: c.aggregable ?? null, dr: c.deltaRange ?? null, cg: c.classGrid };
  });
  expect(cfg.cg).toBe(true);
  expect(cfg.agg).toBeNull();
  expect(cfg.dr).toBeNull();
});

test("place names orient the map: a zoom ladder, optional borders, and a click that goes through", async ({ page }) => {
  // The feature exists for one reason: an SST anomaly off a coastline you can't
  // name tells you nothing about WHERE the ocean is warm. So the test is about
  // legibility — the right number of names at each altitude — and about the
  // thing legibility must not cost: a click still has to reach the globe.
  const sel = page.locator("#places-mode");
  await expect(sel).toHaveValue("labels");   // on by default; a nameless globe is the broken state

  // Only the rungs the camera can actually see get built. Cesium rasterises
  // every glyph into a texture atlas on first draw, so materialising all 7.3k
  // up front costs a ~1.5-second frame on first paint and buys nothing — from
  // orbit you can read about sixty names.
  const loaded = await page.evaluate(async () => {
    const E = window.__earth;
    E.viewer.camera.setView({ destination: Cesium.Cartesian3.fromDegrees(0, 20, 2.4e7) });
    await E.ensureCities();
    return { labels: E.cityLabels.length, points: E.cityPoints.length,
             show: E.cityLabels.show && E.cityPoints.show };
  });
  expect(loaded.labels).toBeGreaterThan(5);
  expect(loaded.labels).toBeLessThan(400);       // a globe view is a handful, not a smear
  expect(loaded.points).toBe(loaded.labels);     // a name with no dot is ambiguous by kilometres
  expect(loaded.show).toBe(true);

  // …and descending the ladder fills in the rest. Building the full set is
  // what the camera hook does on zoom; drive it directly, because the sandbox's
  // software GL makes real camera flights unreliable.
  const all = await page.evaluate(() => window.__earth.buildCitiesTo(99));
  expect(all).toBeGreaterThan(5000);

  // The declutter ladder itself. Natural Earth's `min_zoom` becomes the far end
  // of a per-label DistanceDisplayCondition, so what's on screen is the
  // cartographers' judgement: a handful of world cities from orbit, the whole
  // valley up close. Assert the ladder is MONOTONIC rather than pinning counts,
  // which would break on any Natural Earth re-release.
  const visible = await page.evaluate(() => {
    const L = window.__earth.cityLabels;
    const at = (d) => {
      let n = 0;
      for (let i = 0; i < L.length; i++) {
        const c = L.get(i).distanceDisplayCondition;
        if (d >= c.near && d <= c.far) n++;
      }
      return n;
    };
    return { orbit: at(2.4e7), continent: at(4e6), valley: at(2e5) };
  });
  expect(visible.orbit).toBeGreaterThan(5);
  expect(visible.orbit).toBeLessThan(200);            // more than this is a smear, not a map
  expect(visible.continent).toBeGreaterThan(visible.orbit);
  expect(visible.valley).toBeGreaterThan(visible.continent);
  expect(visible.valley).toBeGreaterThan(5000);       // essentially everything, up close

  // The rung inversion is the one bit of arithmetic here, and it has to agree
  // with the display conditions in both directions or the build lags the view.
  const rung = await page.evaluate(() => {
    const f = window.__earth.cityRungAt;
    return { orbit: f(2.4e7), valley: f(2e5) };
  });
  expect(rung.orbit).toBeLessThan(2);      // from orbit, only Natural Earth's top tier
  expect(rung.valley).toBeGreaterThan(8);  // in the valley, everything it ships

  // A click on a place name must still open the pixel-state card. Label glyphs
  // are far bigger pick targets than they look, and the click handler treats
  // ANY pick as "not bare globe" — so without the CITY_PICK sentinel the app's
  // flagship feature would go quiet in exactly the places the map is most
  // legible. Assert the helper, because canvas click coordinates are unreliable
  // on the software-GL sandbox.
  const through = await page.evaluate(() => {
    const K = window.__earth.CITY_PICK, s = window.__earth.seeThrough;
    return {
      city: s({ id: K }),                                   // a label → invisible to the handler
      real: s({ id: { kind: "station" } })?.id?.kind,        // a real feature → still picked
      none: s(undefined),
    };
  });
  expect(through.city).toBeUndefined();
  expect(through.real).toBe("station");
  expect(through.none).toBeUndefined();

  // "full" adds the GIBS linework as an imagery layer, kept at the top of the
  // stack so switching a data layer on doesn't bury it.
  await sel.selectOption("full");
  const full = await page.evaluate(() => {
    const ils = window.__earth.viewer.imageryLayers, L = window.__earth.bordersLayer;
    return { has: !!L, top: ils.indexOf(L) === ils.length - 1, alpha: L && L.alpha };
  });
  expect(full.has).toBe(true);
  expect(full.top).toBe(true);
  expect(full.alpha).toBeGreaterThan(0.8);   // pale hairlines already; fading them hides them

  await page.evaluate(() => {
    const el = document.querySelector('#layer-list input[data-id="sst"]');
    if (!el.checked) { el.checked = true; el.dispatchEvent(new Event("change", { bubbles: true })); }
  });
  expect(await page.evaluate(() => {
    const ils = window.__earth.viewer.imageryLayers;
    return ils.indexOf(window.__earth.bordersLayer) === ils.length - 1;
  })).toBe(true);   // still on top after a data layer was added

  // "off" is data-only: no names, no dots, no linework — and the choice sticks.
  await sel.selectOption("off");
  const off = await page.evaluate(() => ({
    labels: window.__earth.cityLabels.show,
    points: window.__earth.cityPoints.show,
    borders: !!window.__earth.bordersLayer,
    saved: localStorage.getItem("placesMode"),
  }));
  expect(off.labels).toBe(false);
  expect(off.points).toBe(false);
  expect(off.borders).toBe(false);
  expect(off.saved).toBe("off");
});

test("a place can be searched for, flown to, and read off the map once you get there", async ({ page }) => {
  // The report this feature comes from: a user looking at the sea off Peniche
  // could neither read the town's name nor ask the map where it was. Natural
  // Earth carries twenty-four places in all of Portugal, so BOTH halves needed
  // fixing — the search box, and a deeper tier for it to fly you into.
  await expect(page.locator("#ps-input")).toBeVisible();

  const search = await page.evaluate(async () => {
    const E = window.__earth;
    await E.ensureCities();
    await E.ensureGazetteer();
    const names = (q) => E.searchPlaces(q).map((p) => p.n);
    const peniche = E.searchPlaces("peniche")[0];
    return {
      // the literal complaint
      peniche: peniche && { n: peniche.n, a: peniche.a, o: peniche.o, c: E.placeCountry(peniche) },
      // diacritic-insensitive: an English keyboard must find Zürich
      zurich: names("zurich")[0],
      // both tiers answer through one box — Lisbon is Natural Earth's English
      // exonym, Peniche exists only in the deep file
      lisbon: names("lisbon")[0],
      // an exact name outranks the longer names it is a prefix of
      york: names("york")[0],
      // one letter is not a query; it would return four thousand places
      tooShort: E.searchPlaces("p").length,
      capped: E.searchPlaces("san").length,
    };
  });
  expect(search.peniche).toBeTruthy();
  expect(search.peniche.n).toBe("Peniche");
  expect(search.peniche.a).toBeCloseTo(39.36, 1);
  expect(search.peniche.o).toBeCloseTo(-9.38, 1);
  expect(search.peniche.c).toMatch(/Portugal/);
  expect(search.zurich).toMatch(/^Z(ü|u)rich$/);
  expect(search.lisbon).toMatch(/Lisbo/);
  expect(search.york).toBe("York");
  expect(search.tooShort).toBe(0);
  expect(search.capped).toBeLessThanOrEqual(8);

  // Flying to a place puts you at an altitude derived from the place's own
  // rung, not a fixed number: a hamlet gets a hamlet's altitude and a capital
  // gets a continent's. Assert the ordering, not the metres.
  const heights = await page.evaluate(() => {
    const E = window.__earth;
    const h = (q) => E.placeViewHeight(E.searchPlaces(q)[0]);
    return { peniche: h("peniche"), paris: h("paris") };
  });
  expect(heights.peniche).toBeLessThan(heights.paris);
  expect(heights.peniche).toBeGreaterThan(1e4);

  // The found place gets its own marker, because at any altitude above its own
  // rung the declutter ladder has decided not to draw precisely the place you
  // asked for. Its DOT is unconditional; its NAME is the exact complement of
  // the place's own rung, so arriving somewhere doesn't draw the name twice a
  // pixel apart — which reads as a rendering fault, not as emphasis.
  const marked = await page.evaluate(() => {
    const E = window.__earth;
    const p = E.searchPlaces("peniche")[0];
    E.markFoundPlace(p);
    const l = E.foundLabels.get(0), d = E.foundPoints.get(0);
    return {
      n: E.foundPlace.n, labels: E.foundLabels.length, text: l.text,
      near: l.distanceDisplayCondition.near, far: l.distanceDisplayCondition.far,
      dot: !d.distanceDisplayCondition,
      arrive: E.placeViewHeight(p),
    };
  });
  expect(marked.n).toBe("Peniche");
  expect(marked.labels).toBe(1);
  expect(marked.text).toBe("Peniche");
  expect(marked.dot).toBe(true);                      // the dot never culls
  expect(marked.far).toBeGreaterThan(1e8);            // …and the name reaches orbit
  // The handover: the marker's name stops exactly where the ordinary label
  // starts, and flying there lands you inside the ordinary label's band.
  expect(marked.arrive).toBeLessThan(marked.near);

  // …and arriving there, the deep tier labels the town itself. The camera is
  // driven directly rather than flown: software GL makes real flights
  // unreliable in the sandbox, and this assertion is about the label build.
  const near = await page.evaluate(async () => {
    const E = window.__earth;
    E.viewer.camera.setView({ destination: Cesium.Cartesian3.fromDegrees(-9.38, 39.36, 4e4) });
    // The build is chunked across animation frames and the camera hook may
    // already have started one, so settle rather than assuming one call does it.
    for (let k = 0; k < 20; k++) {
      await E.refreshGazetteerLabels();
      await new Promise(requestAnimationFrame);
      if (E.gazLabels.length) break;
    }
    const L = E.gazLabels, out = [];
    for (let i = 0; i < L.length; i++) out.push(L.get(i).text);
    return { n: L.length, has: out.includes("Peniche"), rung: E.gazData.zFrom };
  });
  expect(near.has, "the deep tier did not label Peniche on arrival").toBe(true);
  expect(near.n).toBeGreaterThan(1);
  expect(near.n).toBeLessThan(1200);   // bounded by the view and the cap, not the file

  // Back at orbit the deep tier must go away entirely — it exists below the
  // rung where Natural Earth runs out, and 54k names on a globe is a smear.
  const orbit = await page.evaluate(async () => {
    const E = window.__earth;
    E.viewer.camera.setView({ destination: Cesium.Cartesian3.fromDegrees(0, 20, 2.4e7) });
    await E.refreshGazetteerLabels();
    await new Promise(requestAnimationFrame);
    return { n: E.gazLabels.length, rungThere: E.gazData.zFrom };
  });
  expect(orbit.n).toBe(0);

  // The box itself: typing renders hits with a disambiguating country, and
  // clearing removes the marker.
  await page.fill("#ps-input", "peniche");
  await expect(page.locator("#ps-results .ps-hit").first()).toContainText("Peniche");
  await expect(page.locator("#ps-results .ps-hit").first()).toContainText("Portugal");
  await page.click("#ps-clear");
  expect(await page.evaluate(() => window.__earth.foundPlace)).toBeNull();
  await expect(page.locator("#ps-results")).toBeHidden();
});

test("islands are named by how wide they are on screen, not by how many people live on them", async ({ page }) => {
  // The report: "can you add island names as well? i think sylt is currently
  // missing". It was missing because both existing tiers are gazetteers of
  // POPULATED PLACES — Westerland (pop. 9,000) was labelled, the 43-km island
  // it stands on was not, because no settlement file contains physical
  // features at all.
  const isl = await page.evaluate(async () => {
    const E = window.__earth;
    E.viewer.camera.setView({ destination: Cesium.Cartesian3.fromDegrees(0, 20, 2.4e7) });
    await E.ensureIslands();
    return {
      count: E.islData.count,
      first: E.islData.islands.slice(0, 3).map((p) => p.n),
      built: E.islLabels.length,
      show: E.islLabels.show,
    };
  });
  expect(isl.count).toBeGreaterThan(2000);
  expect(isl.first[0]).toBe("Greenland");         // the file is an extent ladder
  expect(isl.show).toBe(true);
  // Same prefix-walk economy as the city tier: a glyph atlas is rasterised on
  // first draw, so only the rungs the camera can see are built. From orbit
  // that is the big ones, not five thousand islets.
  expect(isl.built).toBeGreaterThan(10);
  expect(isl.built).toBeLessThan(1000);

  // The rule, which is the whole design: an island earns its name once it is
  // at least as wide on screen as the name is. That makes the threshold a
  // property of the ground and the text, not a hand-picked rung — and it is
  // self-limiting, because filling the view with island names would require
  // islands wider than the view. Assert the monotonicity rather than metres.
  const far = await page.evaluate(() => {
    const E = window.__earth, byName = {};
    for (const p of E.islData.islands) if (!byName[p.n]) byName[p.n] = p;
    const f = (n) => E.islandFar(byName[n]);
    return {
      greenland: f("Greenland"), ireland: f("Ireland"), sylt: f("Sylt"),
      // …and the inverse: the smallest extent worth drawing at a given height
      // must shrink as you descend, or the ladder runs backwards.
      orbit: E.islandExtentAt(2.4e7), valley: E.islandExtentAt(2e5),
    };
  });
  expect(far.greenland).toBeGreaterThan(far.ireland);
  expect(far.ireland).toBeGreaterThan(far.sylt);
  expect(far.greenland).toBeGreaterThan(2e7);   // visible from the full-globe view
  expect(far.sylt).toBeLessThan(1e7);           // …and Sylt is not
  expect(far.valley).toBeLessThan(far.orbit);

  // Descending to the German Bight must actually put the name on the island.
  const bight = await page.evaluate(async () => {
    const E = window.__earth;
    E.viewer.camera.setView({ destination: Cesium.Cartesian3.fromDegrees(8.37, 54.91, 9e5) });
    await E.buildIslandsTo(E.islandExtentAt(9e5));
    const L = E.islLabels, out = [];
    for (let i = 0; i < L.length; i++) {
      const l = L.get(i), c = l.distanceDisplayCondition;
      if (9e5 <= c.far) out.push(l.text);
    }
    return { built: L.length, has: out.includes("Sylt") };
  });
  expect(bight.has, "Sylt is still missing from the globe").toBe(true);
  expect(bight.built).toBeGreaterThan(isl.built);

  // Islands are searchable through the same one box as both settlement tiers —
  // an island is an answer to "where is this", and Sylt is the case that
  // started this.
  const search = await page.evaluate(async () => {
    const E = window.__earth;
    await E.ensureCities();
    await E.ensureGazetteer();
    const s = E.searchPlaces("sylt").find((p) => p.e !== undefined);
    return s && { n: s.n, e: s.e, c: E.placeCountry(s), h: E.placeViewHeight(s) };
  });
  expect(search, "searching for Sylt found no island").toBeTruthy();
  expect(search.n).toBe("Sylt");
  expect(search.c).toMatch(/Germany/);
  // An island has no rung of its own in the file, so the search machinery has
  // to derive one from the same geometry rule — otherwise flying to an island
  // arrives at an altitude where it is not drawn.
  expect(search.h).toBeGreaterThan(1e4);
  expect(search.h).toBeLessThan(search.e * 1000 * 100);

  // Islands follow the places switch like everything else on that control:
  // "off" means a data-only globe, with no furniture of any kind left on it.
  await page.locator("#places-mode").selectOption("off");
  expect(await page.evaluate(() => window.__earth.islLabels.show)).toBe(false);
  await page.locator("#places-mode").selectOption("labels");
  expect(await page.evaluate(() => window.__earth.islLabels.show)).toBe(true);
});

test("every read-out says when its value was observed", async ({ page }) => {
  test.setTimeout(120000);
  // The bug this closes: the app printed a value read from tiles that stop in
  // 2018 or 2022 under a heading saying today's date. gibsTime() has always
  // known the real answer per layer — it just never reached the reader.

  // -- the shared helper resolves each layer at its own granularity ---------
  const w = await page.evaluate(() => {
    const E = window.__earth;
    const cfg = (id) => E.GIBS_LAYERS.find((l) => l.id === id);
    const of = (id) => E.whenOfGibs(cfg(id));
    return {
      today: E.state.date,
      sst: of("sst"),           // daily raster: a day
      grace: of("grace"),       // tiles end 2022-07: a clamped month
      ceres: of("ceres"),       // tiles end 2018-10: a clamped month
      seaice: of("seaice"),     // tiles end 2025-09
      distAnn: of("dist-ann"),  // annual: a year
      nightlights: of("nightlights"),
    };
  });
  // A layer whose tiles have stopped reports the date it really served, and it
  // is emphatically not the date in the selector.
  expect(w.grace).toMatchObject({ kind: "month", t: "2022-07" });
  expect(w.ceres).toMatchObject({ kind: "month", t: "2018-10" });
  // sea ice is a DAILY layer that has stopped, so the clamp shows as a day —
  // the granularity follows the dataset, not the reason the date is old
  expect(w.seaice).toMatchObject({ kind: "day", t: "2025-09-01" });
  expect(w.grace.t).not.toBe(w.today.slice(0, 7));
  // and a live one reports a day, two days back — GIBS's own publication lag
  expect(w.sst.kind).toBe("day");
  expect(Date.parse(w.today) - Date.parse(w.sst.t)).toBeLessThanOrEqual(4 * 864e5);
  expect(w.distAnn.kind).toBe("year");
  expect(w.distAnn.t).toHaveLength(4);

  // -- age: coarsest unit that reads at least 2, never finer than the stamp --
  const ages = await page.evaluate(() => {
    const E = window.__earth, now = Date.parse("2026-08-03T18:55:00Z");
    const a = (kind, t) => E.whenAge({ kind, t }, now);
    return {
      twoDays: a("day", "2026-08-01"),
      sameDay: a("day", "2026-08-03"),
      manyDays: a("day", "2026-06-19"),
      rollUp: a("day", "2026-06-03"),
      oldMonth: a("month", "2022-07"),
      lastYear: a("year", "2025"),
      ahead: a("day", "2026-08-09"),
      fixed: a("period", "1991-2020"),
      label: E.whenLabel({ kind: "period", t: "1991-2020" }),
    };
  });
  expect(ages.twoDays).toBe("2 days old");     // the reading that started this
  expect(ages.sameDay).toBe("today");
  expect(ages.manyDays).toBe("45 days old");   // still days: 1.5 months rounds to nothing useful
  expect(ages.rollUp).toBe("2 months old");    // 61 days does roll up
  expect(ages.oldMonth).toBe("4 years old");
  expect(ages.lastYear).toBe("1 year old");
  expect(ages.ahead).toBe("in 6 days");        // forecast frames read forward
  // A fixed span is not "N years old" — it is simply the years it averages.
  expect(ages.fixed).toBeNull();
  expect(ages.label).toBe("1991–2020");

  // -- baked grids carry their own observation time, read from the file -----
  const grids = await page.evaluate(async () => {
    const E = window.__earth;
    const cfg = (id) => E.GIBS_LAYERS.find((l) => l.id === id);
    const out = {};
    for (const id of ["oisst", "gpcp", "drivers"]) {
      const c = cfg(id);
      out[id] = E.whenOfGrid(c, await E.loadGrid(c));
    }
    const f = cfg("gfs-temp");
    out.forecast = E.whenOfGrid(f, await E.loadGridMonth(f));
    return out;
  });
  expect(grids.oisst).toMatchObject({ kind: "period", t: "1991-2020" });
  expect(grids.gpcp.kind).toBe("period");
  expect(grids.drivers).toMatchObject({ kind: "period", t: "2001-2025" });
  // the forecast grid is day-keyed, so it stamps a day rather than a month
  expect(grids.forecast.kind).toBe("day");
  expect(grids.forecast.t).toHaveLength(10);

  // -- the probe (what a default-state tap actually hits) says it too -------
  // SST is on and the inspector is off, so clicking water runs the probe, not
  // the card. Both must name the same instant; they share one helper so they
  // cannot drift apart.
  const probe = await page.evaluate(async () => {
    const E = window.__earth;
    const res = await E.probeValueAt(Cesium.Cartographic.fromDegrees(8.4, 55.0)); // off Sylt
    const cfg = E.GIBS_LAYERS.find((l) => l.id === "sst");
    return { when: res && res.when, helper: E.whenOfGibs(cfg), title: res && res.title };
  });
  expect(probe.when).toEqual(probe.helper);
  expect(probe.when.kind).toBe("day");

  // and it renders into the tooltip, not just the object
  await page.evaluate(() => window.__earth.probeValueAt(
    Cesium.Cartographic.fromDegrees(8.4, 55.0)).then((r) => window.__earth.renderProbe(r, 40, 40)));
  const vp = page.locator("#value-probe .vp-meta");
  await expect(vp).toContainText(probe.when.t);
  await expect(vp).toContainText(/old|today/);
});

/* The page must still boot with versioned asset URLs, and the manifest and
 * icons it advertises must actually be there. A manifest that 404s is worse
 * than no manifest: the install prompt simply never appears and nothing
 * says why. */
test("the page is installable and serves versioned assets", async ({ page }) => {
  const script = await page.getAttribute('script[src*="app.js"]', "src");
  expect(script).toMatch(/^src\/app\.js\?v=[0-9a-f]{8}$/);
  const css = await page.getAttribute('link[rel="stylesheet"][href*="style.css"]', "href");
  expect(css).toMatch(/^src\/style\.css\?v=[0-9a-f]{8}$/);

  // the stamp on the script tag is the one the About tab shows the user
  const marker = await page.textContent("#build-id");
  expect(script).toContain(marker);

  const href = await page.getAttribute('link[rel="manifest"]', "href");
  expect(href).toMatch(/^manifest\.json\?v=[0-9a-f]{8}$/);

  // fetch it the way the browser would, from the page's own origin
  const manifest = await page.evaluate(async (h) => {
    const r = await fetch(h);
    return r.ok ? await r.json() : { status: r.status };
  }, href);
  expect(manifest.short_name).toBe("Earth");   // capitalised: it is a NAME on a home screen
  expect(manifest.icons.length).toBeGreaterThanOrEqual(3);

  // every icon resolves and decodes to the size the manifest promises
  const icons = await page.evaluate(async (m) => {
    const out = [];
    for (const i of m.icons) {
      const r = await fetch(i.src);
      if (!r.ok) { out.push({ src: i.src, status: r.status }); continue; }
      const bmp = await createImageBitmap(await r.blob());
      out.push({ src: i.src, size: `${bmp.width}x${bmp.height}`, want: i.sizes });
    }
    return out;
  }, manifest);
  for (const i of icons) expect(i.size, `${i.src} did not load`).toBe(i.want);

  // the CSS actually arrived under its versioned URL — a 404 here would leave
  // the app unstyled but still "working", which is easy to miss
  const styled = await page.evaluate(() =>
    getComputedStyle(document.getElementById("sidebar")).position);
  expect(styled).toBe("absolute");
});

test("tide-live layer: paints the globe, moon rides along, tab is the control room", async ({ page }) => {
  const toasts = await recordToasts(page);          // BEFORE the action (see helper)
  await page.click("#tab-tides");                   // opening the tab enables the layer
  await expect(page.locator("#td-clock .stat-value")).not.toHaveText("–", { timeout: 20000 });
  await expect(page.locator("#active-layers")).toContainText("Tide (live)");
  await expect.poll(toasts).toContain("own clock");
  const r = await page.evaluate(() => {
    const tl = window.__earth.tideLive;
    const px = tl.front.getContext("2d").getImageData(0, 0, tl.front.width, tl.front.height).data;
    let painted = 0;
    for (let i = 3; i < px.length; i += 4 * 997) if (px[i] > 0) painted++;
    const fundy = (45 + 90) * 360 + (-66 + 180);
    return {
      painted,
      primShown: tl.prim.show,
      labelCount: tl.labels.length,
      consts: window.__earth.tides.constituents.length,
      playing: window.__earth.tideSim.playing,
      hFundy: window.__earth.tideHeightAt(fundy, Date.now()),
      volume: Number(document.querySelector("#td-volume .stat-value").textContent.replace(/,/g, "")),
      moonLat: window.__earth.tideAstro(Date.now()).moon.lat,
    };
  });
  expect(r.painted).toBeGreaterThan(30);            // ocean cells are actually drawn
  expect(r.primShown).toBe(true);
  expect(r.labelCount).toBe(2);                     // moon + sun
  expect(r.consts).toBe(5);
  expect(r.playing).toBe(true);
  expect(Number.isFinite(r.hFundy)).toBe(true);
  expect(r.volume).toBeGreaterThan(3000);           // planetary-scale km^3
  expect(Math.abs(r.moonLat)).toBeLessThanOrEqual(29);   // lunar standstill band
  // selecting a point (as a globe tap would) draws the 3-day curve in the tab
  const sel = await page.evaluate(() => {
    const ok = window.__earth.tideSelectPoint(
      window.Cesium.Cartographic.fromDegrees(-29.5, 29.5));
    const c = document.getElementById("td-curve");
    const px = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
    let drawn = 0;
    for (let i = 3; i < px.length; i += 4 * 97) if (px[i] > 0) drawn++;
    return { ok, drawn, title: document.getElementById("td-point-title").textContent };
  });
  expect(sel.ok).toBe(true);
  expect(sel.drawn).toBeGreaterThan(5);
  // the CELL's centre, not the tap: a tap near a coast is answered by the
  // nearest water cell, so the label must name the point actually reported
  expect(sel.title).toContain("29.5\u00b0N 29.5\u00b0W");
  // NEXT HIGH / LOW WATER \u2014 the tab's headline answer. The times come from
  // roots of the analytic rate, so they are checked as physics: highs and
  // lows alternate, every high stands above every low, and consecutive
  // turning points are 4\u201326 h apart (semidiurnal ~6.2 h, diurnal ~12.4 h;
  // nothing in between is physical for five constituents).
  const hl = await page.evaluate(() => {
    const E = window.__earth;
    const i = E.tideSim.markCell;
    const t0 = E.tideSim.t;
    const ex = E.tideExtrema(i, t0, 72);
    // the rate really is the height's derivative (finite-difference check)
    const dt = 60000;
    const fd = (E.tideHeightAt(i, t0 + dt) - E.tideHeightAt(i, t0 - dt)) / (2 * dt / 3600000);
    return {
      n: ex.length,
      alternating: ex.every((e, k) => k === 0 || e.high !== ex[k - 1].high),
      gapsOk: ex.every((e, k) => {
        if (!k) return true;
        const h = (e.ms - ex[k - 1].ms) / 3600000;
        return h > 4 && h < 26;
      }),
      minHigh: Math.min(...ex.filter((e) => e.high).map((e) => e.cm)),
      maxLow: Math.max(...ex.filter((e) => !e.high).map((e) => e.cm)),
      // at a turning point the rate must vanish
      rateAtExtremum: Math.abs(E.tideRateAt(i, ex[0].ms)),
      rateErr: Math.abs(E.tideRateAt(i, t0) - fd),
      next: document.getElementById("td-next").textContent,
      emptyHidden: document.getElementById("td-empty").classList.contains("hidden"),
      pointShown: !document.getElementById("td-point").classList.contains("hidden"),
    };
  });
  expect(hl.n).toBeGreaterThanOrEqual(5);          // \u22655 turning points in 3 days
  expect(hl.alternating).toBe(true);
  expect(hl.gapsOk).toBe(true);
  expect(hl.minHigh).toBeGreaterThan(hl.maxLow);   // every high above every low
  expect(hl.rateAtExtremum).toBeLessThan(0.01);    // cm/h \u2014 a true root
  expect(hl.rateErr).toBeLessThan(0.05);           // analytic == finite difference
  expect(hl.next).toContain("next high water");
  expect(hl.next).toContain("next low water");
  expect(hl.next).toMatch(/\d\d:\d\d UTC/);
  expect(hl.next).toMatch(/in \d/);
  expect(hl.emptyHidden).toBe(true);               // invitation gives way to data
  expect(hl.pointShown).toBe(true);
  // "NOW" IS A POINT ON THE CURVE, NOT ITS LEFT EDGE (reported 2026-08-08:
  // you cannot see whether the tide is rising or falling if the line starts
  // at the present moment). The window opens 6 h in the past, so the chart
  // carries at least one turning point that has already happened, and the
  // now-marker sits well inside the canvas.
  const past = await page.evaluate(() => {
    const E = window.__earth;
    const i = E.tideSim.markCell, NOW = E.tideSim.t;
    const c = document.getElementById("td-curve");
    const px = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
    // ink in the leftmost tenth = the past really is drawn, not empty
    let leftInk = 0;
    for (let x = 0; x < c.width / 10; x++) {
      for (let yy = 0; yy < c.height; yy++) if (px[(yy * c.width + x) * 4 + 3] > 0) leftInk++;
    }
    const ex = E.tideExtrema(i, NOW - 6 * 3600000, 78);
    return {
      leftInk,
      pastExtrema: ex.filter((e) => e.ms < NOW).length,
      futureExtrema: ex.filter((e) => e.ms >= NOW).length,
      // where the now-line falls, as a fraction of the width
      nowFrac: 6 / 78,
    };
  });
  expect(past.leftInk).toBeGreaterThan(20);        // the past is painted
  expect(past.pastExtrema).toBeGreaterThanOrEqual(1);   // a turning point behind us
  expect(past.futureExtrema).toBeGreaterThanOrEqual(5);
  expect(past.nowFrac).toBeGreaterThan(0.03);      // interior, not at the edge
  expect(past.nowFrac).toBeLessThan(0.2);          // and the future still owns the chart
  // switching the layer off from its checkbox hides the primitive
  await page.evaluate(() => {
    const el = document.getElementById("toggle-tidelive");
    el.checked = false;
    el.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await expect.poll(() => page.evaluate(() => window.__earth.tideLive.prim.show)).toBe(false);
  // the tagline scene brings it back as the ONLY layer (scenes swap, not pile)
  await page.click('.tag-link[data-scene="tides"]');
  await expect(page.locator("#active-layers .chip:not(.chip-clear)")).toHaveCount(1);
  await expect(page.locator("#active-layers")).toContainText("Tide (live)");
  // the scene also lands you in the control room (the one tab-switching scene)
  await expect(page.locator("#panel-tides")).toBeVisible();
  await expect(page.locator("#tab-tides")).toHaveClass(/active/);
  await expect.poll(() => page.evaluate(() => window.__earth.tideLive.prim.show)).toBe(true);
  // truthful legend on the globe + the sun lights the planet on the sim clock
  await expect(page.locator("#legend-panel")).toContainText("mean sea level");
  const lit = await page.evaluate(() => ({
    lighting: window.__earth.viewer.scene.globe.enableLighting,
    clockSynced: Math.abs(window.Cesium.JulianDate.toDate(
      window.__earth.viewer.clock.currentTime).getTime() - window.__earth.tideSim.t) < 3600e3,
  }));
  expect(lit.lighting).toBe(true);
  expect(lit.clockSynced).toBe(true);
  // off again: lighting returns to the default look
  await page.evaluate(() => {
    const el = document.getElementById("toggle-tidelive");
    el.checked = false;
    el.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await expect.poll(() => page.evaluate(() => window.__earth.viewer.scene.globe.enableLighting)).toBe(false);
});

test("Tides tab answers for wherever you already flew — no blank panel", async ({ page }) => {
  // Reported 2026-08-07: searched Peniche, opened Tides, and found neither
  // the next high/low nor a curve — because the readout only appears after a
  // globe TAP, and the prompt saying so was a grey hint below the fold. Now
  // the tab adopts the camera's point on open, and the coast's land cell
  // resolves to the water next door (the model grid is 1°).
  await page.evaluate(() => window.__earth.viewer.camera.setView({
    destination: Cesium.Cartesian3.fromDegrees(-9.38, 39.36, 300000),   // Peniche
  }));
  await page.click("#tab-tides");
  await expect(page.locator("#td-clock .stat-value")).not.toHaveText("–", { timeout: 20000 });
  await expect(page.locator("#td-point")).toBeVisible();
  await expect(page.locator("#td-empty")).toBeHidden();
  const next = page.locator("#td-next");
  await expect(next).toContainText("next high water");
  await expect(next).toContainText("next low water");
  // a zone label always follows the time — but NOT necessarily "UTC": the
  // point's own zone wins when Open-Meteo answers (Peniche → WEST/GMT+1),
  // the device's zone shows while that is in flight. Assert the shape.
  await expect(next).toHaveText(/\d\d:\d\d\s+\S+/);
  await expect(page.locator("#td-tz")).toContainText("UTC");   // the footer still anchors to UTC
  const r = await page.evaluate(() => {
    const E = window.__earth;
    const d = E.tides;
    const i = E.tideSim.markCell;
    const ix = i % d.nx, iy = Math.floor(i / d.nx);
    const ex = E.tideExtrema(i, E.tideSim.t, 30);
    return {
      lon: d.west + ix + 0.5, lat: d.south + iy + 0.5,
      range: Math.max(...ex.map((e) => e.cm)) - Math.min(...ex.map((e) => e.cm)),
      spring: E.tideSpringRange(i),
      title: document.getElementById("td-point-title").textContent,
    };
  });
  // the adopted cell is the Atlantic just off Peniche, not somewhere else
  expect(Math.abs(r.lon - -9.38)).toBeLessThan(2.5);
  expect(Math.abs(r.lat - 39.36)).toBeLessThan(2.5);
  // and it is a real Portuguese-shelf tide: metres, not centimetres
  expect(r.range).toBeGreaterThan(100);
  expect(r.range).toBeLessThan(800);
  expect(r.title).toMatch(/tide now [-+]?\d+ cm (above|below) mean/);
  // the curve's own vertical range is stated, because the globe's colour
  // scale is fixed at ±2.5 m and the two numbers disagree by design
  expect(r.title).toMatch(/range \d/);
  // and the SPRING range beside it, so a neap window reads as the moon's
  // doing rather than as a broken layer (Peniche: ~2.5 m now, 3.4 m springs)
  expect(r.title).toMatch(/up to \d.* at spring tide/);
  expect(r.spring).toBeGreaterThan(2.5);
  expect(r.spring).toBeLessThan(4.5);
  // times carry a zone, never a bare "16:42": the browser's zone shows
  // instantly, the point's own zone (Open-Meteo timezone=auto) replaces it
  await expect(next).not.toContainText(/\d\d:\d\d\s*<\/span>/);
  const tz = await page.evaluate(() => document.getElementById("td-next").innerText);
  expect(tz).toMatch(/\d\d:\d\d\s+\S/);
  await expect(page.locator("#td-tz")).toContainText("Times are");
});

test("the pixel card reports heat load on a body, not just air temperature", async ({ page }) => {
  // From a Zürich climate-analysis map (PET at 14:00, air temperature at
  // 04:00): what harms people is felt heat by day and the absence of night
  // recovery, neither of which an air-temperature reading shows.
  // The card resolves six live queries before it renders, and the mirror
  // proxy retries transport failures on top of that (scripts/test_proxy.py),
  // so the default 90 s can expire on a slow sandbox and be reported against
  // whichever line was in flight — see CLAUDE.md §4.
  test.setTimeout(180000);
  await page.evaluate(() =>
    window.__earth.showPixelState(Cesium.Cartographic.fromDegrees(8.54, 47.37)));  // Zürich
  const card = page.locator("#pixel-card");
  await expect(card).toContainText("Heat load", { timeout: 60000 });
  await expect(card).toContainText("Feels like now");
  await expect(card).toContainText("vs air");
  await expect(card).toContainText("Felt peak today");
  await expect(card).toContainText("Tonight's low");
  await expect(card).toContainText(/\d tropical night/);
  // the index is NAMED and PET's thresholds are explicitly disclaimed —
  // "feels like" is not PET and must not borrow its 35/41 °C classes
  await expect(card).toContainText("apparent temperature");
  await expect(card).toContainText("PET");
  const t = await page.evaluate(() => document.getElementById("pixel-card").innerText);
  // physically sane for a mid-latitude summer city: felt heat within 15 °C of
  // air, and the tropical-night count is 0..7 of the seven nights fetched
  const felt = /Feels like now\s*([-\d.]+) °C · ([+−][\d.]+) vs air/.exec(t);
  expect(felt).not.toBeNull();
  expect(Math.abs(Number(felt[2].replace("−", "-")))).toBeLessThan(15);
  const trop = /(\d+) tropical night/.exec(t);
  expect(Number(trop[1])).toBeGreaterThanOrEqual(0);
  expect(Number(trop[1])).toBeLessThanOrEqual(7);
});

test("one hanging source cannot stop the pixel card from rendering", async ({ page }) => {
  // Reported 2026-08-16: "the load all data / inspect all data mode is no
  // longer working". Measured against the live site: one climate-api request
  // stayed open for a full minute, and because the card awaits every source
  // before it draws anything, the inspector sat on "Reading this point…" the
  // whole time. fetch() has no timeout, so a connection that never settles
  // used to mean a card that never appears — indistinguishable, from the
  // outside, from a broken app.
  test.setTimeout(240000);
  // A handler that never fulfils, continues OR aborts — the connection just
  // stays open. (A handler that merely returns lets Playwright fail the
  // request immediately, which is a different, much kinder failure.)
  // The MIRROR beforeEach already routes this host to the local proxy, and a
  // handler added on top of it is not reliably the one that answers — with the
  // proxy in play the request FAILS FAST instead of hanging, which is the
  // opposite of the case under test. Drop that route first.
  await page.unroute(/https:\/\/climate-api\.open-meteo\.com\/.*/);
  await page.route(/https:\/\/climate-api\.open-meteo\.com\/.*/,
    () => new Promise(() => {}));
  // fire and forget: showPixelState only resolves once it has given up on the
  // dead host, and the point is what the card shows long before then
  await page.evaluate(() => {
    window.__earth.showPixelState(Cesium.Cartographic.fromDegrees(8.54, 47.37));
  });
  const card = page.locator("#pixel-card");
  // it renders, and with the sections that DID answer. Generous: the deadline
  // is wall-clock, and on the sandbox's software-GL render loop a 15 s timer
  // measured 20-25 s — starved timers, not a slow card.
  await expect(card).toContainText("Air temperature", { timeout: 120000 });
  await expect(card).toContainText("Heat load");
  // the CMIP6 outlook is the section the dead host feeds — absent, and named
  // as still outstanding, because at this moment the app genuinely cannot tell
  // "slow" from "never"
  await expect(card).not.toContainText("Projected change");
  await expect(card).toContainText(/Still loading[^\u2026]*climate outlook/, { timeout: 60000 });
});

test("a slow source is only late, not lost — the card redraws when it lands", async ({ page }) => {
  // The other half of the deadline: a source that is merely slow must not cost
  // its section. The card draws at the deadline with what it has, then redraws
  // complete — so "slow" costs a redraw, never data. Delayed on a baked file
  // rather than a weather host, because those carry no timeout of their own and
  // are therefore the sources that can genuinely still be in flight.
  test.setTimeout(240000);
  const deadline = await page.evaluate(() => window.__earth.PIXEL_DEADLINE_MS);
  expect(deadline).toBeGreaterThan(0);
  await page.route(/ocean_column\.json/, async (route) => {
    // Comfortably past the deadline: the draw is fired by a wall-clock timer,
    // and on the sandbox's starved render loop a 15 s timer has been measured
    // firing at 20-25 s. A margin under that drift would let the "straggler"
    // arrive before the first draw, and the test would prove nothing.
    await new Promise((r) => setTimeout(r, deadline + 20000));   // late, but it arrives
    await route.continue();
  });
  // Fire and FORGET — showPixelState resolves only after its second pass, so
  // awaiting it here would skip past the very state under test. The assertions
  // below watch the card the way the user does: incomplete, then filled in.
  await page.evaluate(() => {
    window.__earth.showPixelState(Cesium.Cartographic.fromDegrees(-30, 40));
  });
  const card = page.locator("#pixel-card");
  // first pass: drawn without it, and honest about why
  await expect(card).toContainText(/Still loading[^\u2026]*ocean column/, { timeout: 90000 });
  // second pass: the straggler lands, its section appears, the notice is gone
  await expect(card).toContainText("Ocean column", { timeout: 90000 });
  await expect(card).not.toContainText("Still loading");
});

test("a straggler cannot redraw the card under a newer point", async ({ page }) => {
  // The second pass makes this reachable: tap A, tap B while A's slow source is
  // still out, and A's late data would arrive to find the card headed "B".
  test.setTimeout(240000);
  const deadline = await page.evaluate(() => window.__earth.PIXEL_DEADLINE_MS);
  await page.route(/ocean_column\.json/, async (route) => {
    await new Promise((r) => setTimeout(r, deadline + 20000));
    await route.continue();
  });
  await page.evaluate(() => {
    window.__earth.showPixelState(Cesium.Cartographic.fromDegrees(-30, 40));   // A, ocean
  });
  const card = page.locator("#pixel-card");
  await expect(card).toContainText("40.00\u00b0N 30.00\u00b0W", { timeout: 90000 });
  // B, before A's straggler lands
  await page.evaluate(() => {
    window.__earth.showPixelState(Cesium.Cartographic.fromDegrees(8.54, 47.37));
  });
  await expect(card).toContainText("47.37\u00b0N 8.54\u00b0E", { timeout: 90000 });
  // long enough for A's second pass to have fired
  await page.waitForTimeout(deadline + 30000);
  await expect(card).toContainText("47.37\u00b0N 8.54\u00b0E");
  await expect(card).not.toContainText("40.00\u00b0N 30.00\u00b0W");
  // and B's own content, not A's ocean rows, is what's in the body
  await expect(card).toContainText("Heat load");
});

test("searching a place points the tide dashboard at it, by name", async ({ page }) => {
  // Reported 2026-08-07: "if I search for a place, the left panel should show
  // data for that place (without clicking on the globe)" — and should say
  // which place it is showing.
  await page.evaluate(async () => {
    const E = window.__earth;
    await E.ensureCities();
    await E.ensureGazetteer();
  });
  await page.click("#tab-tides");
  await expect(page.locator("#td-clock .stat-value")).not.toHaveText("–", { timeout: 20000 });
  // search the way a user does: type, then take the first result
  await page.fill("#ps-input", "peniche");
  await page.click("#ps-results li >> nth=0");
  const title = page.locator("#td-point-title");
  await expect(title).toContainText("Peniche");
  await expect(title).toContainText("tide now");
  await expect(page.locator("#td-empty")).toBeHidden();
  await expect(page.locator("#td-next")).toContainText("next high water");
  // and the point really is Peniche's water, not the last cell selected
  const near = await page.evaluate(() => {
    const E = window.__earth, d = E.tides, i = E.tideSim.markCell;
    return { lon: d.west + (i % d.nx) + 0.5, lat: d.south + Math.floor(i / d.nx) + 0.5,
             place: E.tideSim.markPlace };
  });
  expect(near.place).toBe("Peniche");
  expect(Math.abs(near.lon - -9.38)).toBeLessThan(2.5);
  expect(Math.abs(near.lat - 39.36)).toBeLessThan(2.5);
});

test("with the tide on, a globe tap reads its height like any other layer", async ({ page }) => {
  // Reported 2026-08-07: the tide answered only in its own tab — the standard
  // read-out and the source-cell marker every other layer shows did not fire.
  await page.check("#toggle-tidelive");
  await expect.poll(() => page.evaluate(() => !!window.__earth.tides), { timeout: 20000 }).toBe(true);
  const r = await page.evaluate(async () => {
    const E = window.__earth;
    const res = await E.probeValueAt(Cesium.Cartographic.fromDegrees(-29.5, 29.5));
    const land = await E.probeValueAt(Cesium.Cartographic.fromDegrees(0, 20));   // Sahara
    return { res, land };
  });
  expect(r.res.title).toContain("Tide height");
  expect(r.res.units).toBe("m");
  expect(Math.abs(r.res.value)).toBeLessThan(2.5);        // metres, not centimetres
  expect(r.res.extra).toMatch(/next (high|low) water/);   // the probe names it too
  expect(r.res.cell.east - r.res.cell.west).toBeCloseTo(1, 5);   // the 1° model cell
  // inland the tide has nothing to say, so the probe falls THROUGH to the
  // layer underneath (SST, which answers "no data" there) rather than
  // claiming a tide in the Sahara
  expect(r.land === null || !String(r.land.title).includes("Tide")).toBe(true);
  // the full click path: read-out visible, source cell outlined on the globe
  await page.evaluate(() => {
    const c = window.__earth.viewer.scene.canvas;
    return window.__runProbe(c.clientWidth / 2, c.clientHeight / 2, true);
  });
  const shown = await page.evaluate(() => ({
    probe: !document.getElementById("value-probe").classList.contains("hidden"),
    text: document.getElementById("value-probe").innerText,
    mark: window.__earth.probeMark?.dot.show,
    cellOutlined: window.__earth.probeMark?.edge.show,
  }));
  if (shown.probe) {                       // centre of view may be off the globe
    expect(shown.text).toMatch(/Tide height|°/);
    expect(shown.mark).toBe(true);
    expect(shown.cellOutlined).toBe(true);
  }
});

test("tidal-range layer: chip, dateless toast, probe-able grid", async ({ page }) => {
  const toasts = await recordToasts(page);          // BEFORE the action (see helper)
  await page.evaluate(() => {
    // GIBS_LAYERS rows carry data-id, not toggle-<id> (that's hand-written layers)
    const el = document.querySelector('input[data-id="tides"]');
    el.checked = true;
    el.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await expect(page.locator("#active-layers")).toContainText("Tidal range");
  await expect.poll(toasts).toContain("fixed harmonic analysis");
  const g = await page.evaluate(async () => {
    const cfg = window.__earth.GIBS_LAYERS.find((l) => l.id === "tides");
    const grid = await (await fetch(cfg.gridFile)).json();
    const cell = (lon, lat) => grid.values[(Math.floor(lat) + 90) * 360 + (Math.floor(lon) + 180)];
    return { units: grid.units, fundy: cell(-65.5, 45.5), land: cell(15.5, 47.5) };
  });
  expect(g.units).toBe("m");
  expect(g.fundy).toBeGreaterThan(5);
  expect(g.land).toBeNull();                        // Austria has no tide
});

test("installed-app update check: a newer served build offers a one-tap reload", async ({ page }) => {
  const toasts = await recordToasts(page);
  await page.route(/index\.html\?fresh=/, (route) => route.fulfill({
    contentType: "text/html",
    body: '<script src="src/app.js?v=ffffffff"></script>',
  }));
  const served = await page.evaluate(() => window.__earth.checkForNewBuild());
  expect(served).toBe("ffffffff");
  await expect.poll(toasts).toContain("newer build of earth");
  await expect(page.locator("#reload-now")).toBeVisible();
  // same-build response: no new toast, returns the matching stamp
  await page.unroute(/index\.html\?fresh=/);
  const same = await page.evaluate(async () => {
    const ours = document.querySelector('script[src*="src/app.js?v="]').getAttribute("src").split("v=")[1];
    return (await window.__earth.checkForNewBuild()) === ours;
  });
  expect(same).toBe(true);
});


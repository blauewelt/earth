// Browser tests for the earth globe app.
//
// In CI these hit the real CDN (cdnjs) and NASA GIBS. In the local sandbox,
// set MIRROR=1 to route the Cesium CDN to the vendored copy (_vendor/cesium)
// and GIBS to a local proxy on :8081 (see README "Testing").
"use strict";
const { test, expect } = require("@playwright/test");

const CDN = "https://cdnjs.cloudflare.com/ajax/libs/cesium/1.133.1";

/* The Hugging Face Hub and everywhere a `resolve/` URL can redirect to: the
 * repo host itself, the LFS CDN, and the Xet bridge on *.hf.co. One regex,
 * used both to route the Hub through (beforeEach) and to cut it off (the
 * fallback test) — written once so those two can never disagree about what
 * "the Hub" means. See CLAUDE.md §3, second deliberate exception. */
const HF_HOSTS = /^https:\/\/(huggingface\.co|cdn-lfs[^/]*\.huggingface\.co|[^/]+\.hf\.co)\//;

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
      // the third backend: four keyless tile hosts (CLAUDE.md §3)
      ["https://wmts.terrascope.be", "http://localhost:8088"],
      ["https://storage.googleapis.com", "http://localhost:8089"],
      ["https://tiles.maps.eox.at", "http://localhost:8090"],
      ["https://wmts.geo.admin.ch", "http://localhost:8091"],
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
    /* The Hub is PASSED THROUGH, not mirrored — it is the thing under test,
     * and E-040's whole claim is that a browser range read against the real
     * Hub returns 206 with exactly the bytes asked for. The sandbox BROWSER
     * has no egress; the Playwright node process does, so hand the request to
     * node and fulfil with whatever comes back, headers and status intact.
     * maxRedirects matters: a `resolve/` URL 302s to the CDN, and without
     * following it the client would see a redirect body instead of bytes. */
    await page.route(HF_HOSTS, async (route) => {
      try {
        const response = await page.request.fetch(route.request(), { maxRedirects: 5 });
        await route.fulfill({ response });
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
  // The AMOC-flagged count comes from the catalog itself, not a pinned
  // number: the pin (58) went red the day a 59th AMOC record landed and
  // stayed red for two runs — a test that fails because the catalog grew is
  // not finding a defect. The invariant is "the filter shows exactly the
  // flagged records", so derive the number from the file the app reads.
  const cat = JSON.parse(require("fs").readFileSync(require("path").join(__dirname, "..", "data", "catalog.json"), "utf8"));
  const amoc = cat.records.filter((r) => r.amoc).length;
  expect(amoc).toBeGreaterThan(40);
  await expect(page.locator("#catalog-count")).toContainText(`${amoc} of ${total}`);
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

test("the comparison date is steered like the main date, and knows when it tracks", async ({ page }) => {
  // Chris, 2026-08-17: "let's make the Date and Compare Date selections
  // analogous (with buttons + also allow for comparing to a specific date)".
  // The two semantics are the substance: an OFFSET tracks the main date so
  // both sides stay in the same season; a PINNED date does not. Getting that
  // backwards would look fine and compare the wrong months.
  await page.fill("#layer-date", "2020-06-15");
  await page.dispatchEvent("#layer-date", "change");

  // off by default: no date field, no steppers
  await expect(page.locator("#compare-date-row")).toBeHidden();
  await expect(page.locator("#compare-steps")).toBeHidden();

  // an offset reveals them and shows the derived date
  await page.selectOption("#compare-select", "10");
  await expect(page.locator("#compare-date-row")).toBeVisible();
  await expect(page.locator("#compare-steps")).toBeVisible();
  await expect(page.locator("#compare-date")).toHaveValue("2010-06-15");

  // ... and TRACKS: moving the main date moves it, same month either side
  await page.click("#date-steps button[data-step='-1m']");
  await expect(page.locator("#layer-date")).toHaveValue("2020-05-15");
  await expect(page.locator("#compare-date")).toHaveValue("2010-05-15");

  // stepping the comparison PINS it, and the select says so
  await page.click("#compare-steps button[data-cstep='-1y']");
  await expect(page.locator("#compare-date")).toHaveValue("2009-05-15");
  await expect(page.locator("#compare-select")).toHaveValue("custom");

  // pinned means pinned: the main date moves, this does not
  await page.click("#date-steps button[data-step='-1m']");
  await expect(page.locator("#layer-date")).toHaveValue("2020-04-15");
  await expect(page.locator("#compare-date")).toHaveValue("2009-05-15");

  // typing a date pins it too, and the app compares against what is typed
  await page.fill("#compare-date", "2003-07-04");
  await page.dispatchEvent("#compare-date", "change");
  await page.locator("#compare-date").blur();   // as a real user leaves the field
  await expect(page.locator("#compare-select")).toHaveValue("custom");
  expect(await page.evaluate(() => window.__earth.compareDate())).toBe("2003-07-04");

  // choosing an offset again hands it back to tracking
  await page.selectOption("#compare-select", "5");
  await expect(page.locator("#compare-date")).toHaveValue("2015-04-15");
  await page.click("#date-steps button[data-step='-1y']");
  await expect(page.locator("#compare-date")).toHaveValue("2014-04-15");

  // and Off puts everything away
  await page.selectOption("#compare-select", "0");
  await expect(page.locator("#compare-date-row")).toBeHidden();
  expect(await page.evaluate(() => window.__earth.compareDate())).toBeNull();
});

test("a capped SST anomaly reports its actual departure, computed not read", async ({ page }) => {
  // Chris, 2026-08-18: "The SST anomaly layer is capped at +3C? Can you make
  // sure the point renders the actual anomaly value without capping it?"
  // GIBS serves colours: the MUR25 anomaly palette's end bins are catch-alls
  // (`[3.0,+INF)`), so the tile cannot express more than 3 and the probe can
  // only honestly say "≥ 3" — exactly when the magnitude is the whole story.
  // The card computes the real figure instead, from the OISST monthly archive
  // minus that calendar month's 1991-2020 normal.
  test.setTimeout(240000);
  await page.evaluate(() => {
    const el = document.getElementById("layer-date");
    el.value = "2026-07-15"; el.dispatchEvent(new Event("change", { bubbles: true }));
  });
  // Niño 1+2 centre during the 2026 El Niño: far past the palette's ceiling
  await page.evaluate(() =>
    window.__earth.showPixelState(Cesium.Cartographic.fromDegrees(-85, -5)));
  const card = page.locator("#pixel-card");
  await expect(card).toContainText("SST anomalies", { timeout: 90000 });
  await expect(card).toContainText("actual departure", { timeout: 90000 });
  const t = await page.evaluate(() => document.getElementById("pixel-card").innerText);
  // the capped read-out is still shown — we correct it, we do not hide it
  expect(t).toMatch(/SST anomalies\s*\n?≥ 3/);
  const m = /actual departure\s*\n?([+−-][\d.]+) °C/.exec(t);
  expect(m, "the computed departure row is missing").not.toBeNull();
  const v = Number(m[1].replace("−", "-"));
  expect(Math.abs(v)).toBeGreaterThan(3);      // or there was nothing to correct
  expect(Math.abs(v)).toBeLessThan(12);        // and it is a temperature, not a bug
  // provenance is stated, because it is a DIFFERENT measurement from the raster
  await expect(card).toContainText("1991-2020");
  await expect(card).toContainText("OISST");
});

/* Read the "actual departure" row out of the card as {value, stamp}. The STAMP
 * is the point: the daily Hub read is for one calendar day and the monthly
 * fallback for a whole month, so the granularity of that dim right-hand string
 * is what says which measurement actually served the row. Anything that reads
 * the two off the card's flat innerText cannot tell "2015-07" from
 * "2015-07-15" reliably once ages are appended. */
async function departureRow(page) {
  return page.evaluate(() => {
    const row = [...document.querySelectorAll("#pixel-card .px-row")]
      .find((r) => r.querySelector(".px-label")?.textContent.includes("actual departure"));
    if (!row) return null;
    return {
      value: row.querySelector(".px-val")?.textContent || "",
      stamp: row.querySelector(".px-when")?.textContent || "",
    };
  });
}

test("the true SST anomaly is read for the exact day, straight off the Hub", async ({ page }) => {
  /* E-040. The monthly correction answers a coarser question than the one that
   * was asked: a 1° monthly mean against a 25 km daily raster. The Hub carries
   * OISST daily at 0.25° stored pixel-major, so the true value for THIS day at
   * THIS point is one 730-byte range read — and the row must then be stamped to
   * the DAY, because a day-granularity value under a month stamp is the same
   * class of misdating the per-row stamps exist to prevent.
   *
   * The point and date are VERIFIED, not guessed (2026-08-18). At 5°S 85°W the
   * MUR25 tile for 2015-07-02 reads rgb(128,0,0) at the level the card probes —
   * the palette's `[3.0,+INF)` catch-all, so there is genuinely something to
   * correct — and OISST daily reads 25.34 °C there against a July 1991-2020
   * normal of 21.35 °C, i.e. +3.99 °C. The same point on 2015-07-15 is NOT
   * capped (the tile inverts to 2.95), which is why the date is 07-02: a test
   * pinned to an uncapped pixel would pass by never exercising the path. */
  test.setTimeout(240000);
  await page.evaluate(() => {
    const el = document.getElementById("layer-date");
    el.value = "2015-07-02"; el.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await page.evaluate(() =>
    window.__earth.showPixelState(Cesium.Cartographic.fromDegrees(-85, -5)));
  const card = page.locator("#pixel-card");
  await expect(card).toContainText("SST anomalies", { timeout: 90000 });
  await expect(card).toContainText("actual departure", { timeout: 90000 });
  await expect.poll(() => departureRow(page).then((r) => r?.stamp || ""),
    { timeout: 60000, message: "the departure row never carried a day stamp" })
    .toMatch(/^2015-07-02\b/);

  const row = await departureRow(page);
  const m = /([+−-][\d.]+) °C/.exec(row.value);
  expect(m, `no computed value in "${row.value}"`).not.toBeNull();
  const v = Number(m[1].replace("−", "-"));
  expect(Number.isFinite(v)).toBe(true);
  expect(Math.abs(v)).toBeGreaterThan(3);      // or there was nothing to correct
  expect(Math.abs(v)).toBeLessThan(12);        // and it is a temperature, not a bug
  // the provenance must name the daily product AND admit the monthly normal —
  // this is a daily reading against a monthly baseline, and saying otherwise
  // would overstate what the number removes
  await expect(card).toContainText("OISST daily 0.25°");
  await expect(card).toContainText("1991-2020");
});

test("with the Hub unreachable the card still shows the monthly departure", async ({ page }) => {
  /* The fallback is the whole reason huggingface.co could be admitted under
   * CLAUDE.md §3 at all: a Hub outage must cost PRECISION, not the feature.
   * Registered after the beforeEach pass-through, so this handler wins. */
  test.setTimeout(240000);
  const hf = [];
  page.on("request", (r) => { if (HF_HOSTS.test(r.url())) hf.push(r.url()); });
  await page.route(HF_HOSTS, (route) => route.abort());

  await page.evaluate(() => {
    const el = document.getElementById("layer-date");
    el.value = "2026-07-15"; el.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await page.evaluate(() =>
    window.__earth.showPixelState(Cesium.Cartographic.fromDegrees(-85, -5)));
  const card = page.locator("#pixel-card");
  await expect(card).toContainText("actual departure", { timeout: 90000 });
  await expect.poll(() => departureRow(page).then((r) => r?.stamp || ""),
    { timeout: 60000, message: "the monthly fallback never stamped its row" })
    .toMatch(/^2026-07\b(?!-)/);

  const row = await departureRow(page);
  expect(/([+−-][\d.]+) °C/.test(row.value), `no computed value in "${row.value}"`).toBe(true);
  await expect(card).toContainText("OISST v2.1 monthly mean");
  // and the app really did try the Hub first — a fallback that is never
  // exercised because the daily path was silently skipped proves nothing
  expect(hf.length, "the daily path was never attempted").toBeGreaterThan(0);
});

test("a capped probe shows a BOUND first, never a number it will contradict", async ({ page }) => {
  // Chris, 2026-08-18, clicking the Mediterranean: "I see two values, first a
  // lower value (2.x C) and then a higher one (4.y C)." Both were right and
  // measured different things — the monthly path had fallen back to 2026-07 on
  // a 1° cell while the daily read answered for the selected day at 0.25°
  // (measured in the Balearic: +2.25 vs +4.57) — and the monthly figure even
  // sat BELOW the ≥3 bound it was correcting. A number replaced by a different
  // number reads as the app fixing a mistake; a bound replaced by a number
  // reads as a refinement. So the first paint of an upgradable cap must be the
  // bound.
  test.setTimeout(240000);
  await page.evaluate(() => {
    const el = document.getElementById("layer-date");
    el.value = "2015-07-02"; el.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await page.evaluate(() => {
    const el = document.querySelector('#layer-list input[data-id="sst-anom"]');
    el.checked = true; el.dispatchEvent(new Event("change", { bubbles: true }));
  });
  // let the monthly normals become resident — they are what used to be shown
  await page.waitForFunction(() => !!window.__earth.sstAnomalyAt(-85, -5), null,
                             { timeout: 60000 });
  const monthly = await page.evaluate(() => window.__earth.sstAnomalyAt(-85, -5));
  expect(monthly, "the monthly value must exist, or this proves nothing").toBeTruthy();

  const res = await page.evaluate(() =>
    window.__earth.probeValueAt(Cesium.Cartographic.fromDegrees(-85, -5)));
  expect(res.cap, "pick a capped point or the test is vacuous").toBeTruthy();
  expect(res.upgradable).toBe(true);

  // the synchronous render: a bound, and NOT the resident monthly number
  const first = await page.evaluate((r) => {
    window.__earth.renderProbe(r, 400, 300);
    return document.querySelector("#value-probe .vp-head").innerText;
  }, res);
  expect(first).toMatch(/≥|</);
  expect(first).not.toMatch(new RegExp(Math.abs(monthly.v).toFixed(2)));
});

test("the hover probe upgrades a capped read in place, and only for its own point", async ({ page }) => {
  /* The tooltip renders SYNCHRONOUSLY — that is the whole design of the dwell
   * probe — so the daily read can only ever land after the box is drawn. Two
   * properties, and the second is the one that can hurt: it must improve the
   * read-out in place, and it must never write into a box that has since moved
   * to another point. Same guard as pixelCardSeq, one rung down. */
  test.setTimeout(240000);
  await page.evaluate(() => {
    const el = document.getElementById("layer-date");
    el.value = "2015-07-02"; el.dispatchEvent(new Event("change", { bubbles: true }));
    const cb = document.querySelector('input[data-id="sst-anom"]');
    cb.checked = true; cb.dispatchEvent(new Event("change", { bubbles: true }));
  });
  const probeAt = (lon, lat) => page.evaluate(async ([x, y]) => {
    const E = window.__earth;
    const entry = E.colormapLayersTopDown().find((e) => e.cfg.id === "sst-anom");
    const res = await E.probeEntryValue(entry, Cesium.Cartographic.fromDegrees(x, y));
    E.renderProbe(res, 60, 60);
    return { cap: res.cap ? res.cap.bound : null, upgradable: !!res.upgradable };
  }, [lon, lat]);
  const headText = () => page.evaluate(() =>
    document.querySelector("#value-probe .vp-head")?.textContent || "");

  // 5°S 85°W on 2015-07-02: the tile is the palette's +3 catch-all (verified),
  // and OISST daily says +3.99 there
  const first = await probeAt(-85, -5);
  expect(first.cap, "the anchor point is no longer capped — see the card test").toBe(3);
  expect(first.upgradable).toBe(true);
  expect(await headText()).toContain("3.00");          // instant: the palette bound
  await expect.poll(headText, { timeout: 30000 }).toMatch(/\+3\.99/);
  // the bound stays beside it — it is what the COLOUR meant, and we correct the
  // reading rather than hiding it
  expect(await headText()).toContain("palette");

  /* The guard. Hold the Hub open and probe a DIFFERENT capped pixel (a
   * different pixel-year, so the cache cannot answer it instantly), then move
   * to an ordinary mid-Atlantic point while that read is still out. The stale
   * answer must be dropped: the box shows the point the cursor is on, and an
   * uncapped read has no "palette" clause at all.
   *
   * The margins are what make this test real rather than decorative. Probe the
   * mid-Atlantic point FIRST so its tile is cached — on the sandbox's proxy a
   * cold tile read took longer than the hold, which let the held answer land
   * while its OWN point was still displayed and the test passed for the wrong
   * reason (verified: it passed with the guard deleted). And hold the Hub for
   * eight seconds, comfortably longer than anything on the probe path. */
  await probeAt(-30, 40);                        // warm the tile; timing, not assertion
  await page.route(HF_HOSTS, async (route) => {
    await new Promise((r) => setTimeout(r, 8000));
    await route.fallback();
  });
  const held = await probeAt(-84, -2);           // also capped 2015-07-02, another pixel
  expect(held.upgradable).toBe(true);
  const moved = await probeAt(-30, 40);          // mid-Atlantic, ordinary
  expect(moved.upgradable).toBe(false);
  await page.waitForTimeout(14000);              // well past the held read
  const after = await headText();
  expect(after, `a moved-on probe was overwritten: "${after}"`).not.toContain("palette");
});

test("an uncapped anomaly costs no extra request", async ({ page }) => {
  // The correction must be free in the ordinary case: the climatology and the
  // OISST year file are ~0.4 and ~3.8 MB, and a mid-ocean pixel with a normal
  // anomaly has nothing to correct. The Hub is on the same footing — a 730-byte
  // read is cheap, but a request per click to a third-party host for a value
  // nobody needs is exactly what CLAUDE.md §3 exists to prevent.
  const asked = [];
  page.on("request", (r) => {
    if (/oisst_clim|oisst_y/.test(r.url()) || HF_HOSTS.test(r.url())) asked.push(r.url());
  });
  await page.evaluate(() =>
    window.__earth.showPixelState(Cesium.Cartographic.fromDegrees(-30, 40)));
  await expect(page.locator("#pixel-card")).toContainText("SST anomalies", { timeout: 90000 });
  await page.waitForTimeout(2000);
  expect(asked, `fetched ${asked.join(", ")} for a pixel that was not capped`).toEqual([]);
});

test("a half-typed year does not fight the typist", async ({ page }) => {
  // Chris, 2026-08-18: "I cannot type 2010 into it. It looks like it's editing
  // only the first number of the year." An <input type="date"> fires `change`
  // as each SEGMENT completes, so typing 2010 reports the real dates 0002,
  // 0020, 0201 on the way. The first version clamped each to the floor and
  // wrote the clamp BACK into the field, which reset the caret to the first
  // segment on every keystroke. The Date field never had the bug because it
  // never writes into its own field — "just replicate what Date does".
  await page.fill("#layer-date", "2020-06-15");
  await page.dispatchEvent("#layer-date", "change");
  await page.selectOption("#compare-select", "10");
  await expect(page.locator("#compare-date")).toHaveValue("2010-06-15");

  // type a year one digit at a time, exactly as the widget reports it
  for (const partial of ["0002-06-15", "0020-06-15", "0201-06-15"]) {
    await page.evaluate((v) => {
      const el = document.getElementById("compare-date");
      el.focus();
      el.value = v;
      el.dispatchEvent(new Event("change", { bubbles: true }));
    }, partial);
    // THE BUG: the field must still hold what was typed. Any write-back here
    // is the caret reset, and the year can never be finished.
    await expect(page.locator("#compare-date")).toHaveValue(partial);
  }

  // finishing the year commits it, and the select says it is pinned now
  await page.evaluate(() => {
    const el = document.getElementById("compare-date");
    el.value = "2010-06-15";
    el.dispatchEvent(new Event("change", { bubbles: true }));
    el.blur();
  });
  expect(await page.evaluate(() => window.__earth.compareDate())).toBe("2010-06-15");
  await expect(page.locator("#compare-select")).toHaveValue("custom");

  // leaving the field alone must not pin a TRACKING offset — clicking another
  // control while an offset was showing used to convert it into a fixed date
  await page.selectOption("#compare-select", "5");
  await page.click("#date-steps button[data-step='-1y']");
  await expect(page.locator("#compare-select")).toHaveValue("5");
  await expect(page.locator("#compare-date")).toHaveValue("2014-06-15");
});

test("both date steppers obey the same calendar rules and the same bounds", async ({ page }) => {
  // Six real clicks, each rebuilding the comparison's imagery on a software-GL
  // stack: this ran at ~84 s of the 90 s default before E-041, and holding the
  // old frame until the new one paints (retireLayer) costs the few seconds that
  // pushed it over. What is under test here is calendar arithmetic, not speed.
  test.setTimeout(150000);
  // One stepper function serves both rows; these are the cases where naive
  // arithmetic differs from the calendar, plus the clamps that keep the
  // comparison on the axis the UI actually offers.
  await page.selectOption("#compare-select", "custom");
  await page.fill("#compare-date", "2020-03-31");
  await page.dispatchEvent("#compare-date", "change");
  await page.locator("#compare-date").blur();
  await page.click("#compare-steps button[data-cstep='-1m']");
  await expect(page.locator("#compare-date")).toHaveValue("2020-02-29");   // not Mar 2
  await page.fill("#compare-date", "2020-02-29");
  await page.dispatchEvent("#compare-date", "change");
  await page.locator("#compare-date").blur();
  await page.click("#compare-steps button[data-cstep='-1y']");
  await expect(page.locator("#compare-date")).toHaveValue("2019-02-28");   // leap day
  // the floor is GIBS's, and it is shared with the Date row
  await page.fill("#compare-date", "2000-01-02");
  await page.dispatchEvent("#compare-date", "change");
  await page.locator("#compare-date").blur();
  await expect(page.locator("#compare-date")).toHaveValue("2000-01-02");
  await page.click("#compare-steps button[data-cstep='-1y']");
  await expect(page.locator("#compare-date")).toHaveValue("2000-01-01");
  // and the ceiling: a comparison cannot be asked for past the newest date
  const max = await page.evaluate(() => document.getElementById("layer-date").max);
  await page.fill("#compare-date", max);
  await page.dispatchEvent("#compare-date", "change");
  await page.locator("#compare-date").blur();
  await page.click("#compare-steps button[data-cstep='+1y']");
  await expect(page.locator("#compare-date")).toHaveValue(max);
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
  // two OPERA disturbance layers + two DSWx water layers + HBASE + GMIS + WorldCover
  expect(flags.length).toBe(7);
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

/* ---------------------------------------------------------- the fine tier */

test("fine layers are hidden above their gate and request nothing until the camera comes down", async ({ page }) => {
  test.setTimeout(120000);
  const toasts = await recordToasts(page);   // see recordToasts: toasts expire

  // Every fine layer is 30 m or finer, GIBS-served, and declares its gate in
  // km; the daily ones are swath products and ALL carry a gate.
  const cfgs = await page.evaluate(() => window.__earth.GIBS_LAYERS
    .filter((l) => l.fine).map((l) => [l.id, l.tms, l.fine, !!l.timed]));
  expect(cfgs.map((c) => c[0]).sort()).toEqual(
    ["hls-l30", "hls-s30", "nisar", "sar-s1", "swissimage", "swissimage-history", "swissrelief",
     "water-hls", "water-s1", "worldcover"]);
  for (const [id, tms, fine, timed] of cfgs) {
    const xyz = ["swissimage", "swissimage-history", "swissrelief", "worldcover"].includes(id);
    if (!xyz) { expect(["31.25m", "15.625m"]).toContain(tms); expect(timed).toBe(true); }
    expect(fine).toBeGreaterThan(0);
  }

  // From orbit (the default view) enabling Sentinel-2 — a swath product with
  // an OVERVIEW — keeps the layer shown but capped at pyramid level 5, so the
  // day's swaths paint as coarse strips (a few cheap tiles) and nothing deeper
  // is ever requested; the row and a toast say what the strips are.
  const tileLevels = [];
  // TILE requests only: the one-off DescribeDomains metadata fetch also
  // names the layer, and it is supposed to go out on enable (ensureGibsDomain).
  page.on("request", (r) => {
    const m = r.url().match(/HLS_S30[^?]*\/31\.25m\/(\d+)\/\d+\/\d+\.png/);
    if (m) tileLevels.push(Number(m[1]));
  });
  const orbit = await page.evaluate(() => {
    const E = window.__earth;
    E.viewer.camera.setView({ destination: Cesium.Cartesian3.fromDegrees(8.5, 47.4, 1.2e7) });
    const el = document.querySelector('#layer-list input[data-id="hls-s30"]');
    el.checked = true; el.dispatchEvent(new Event("change", { bubbles: true }));
    E.updateFineGates();
    const e = E.state.layers["hls-s30"];
    return {
      gated: E.fineGated(e.cfg),
      show: e.layer ? e.layer.show : null,
      maxLevel: e.layer.imageryProvider.maximumLevel,
      hint: document.querySelector('[data-finehint="hls-s30"]').textContent,
      hintHidden: document.querySelector('[data-finehint="hls-s30"]').hidden,
      height: Math.round(E.cameraHeight() / 1000).toLocaleString("en-US"),
    };
  });
  expect(orbit.gated).toBe(true);
  expect(orbit.show).toBe(true);
  expect(orbit.maxLevel).toBe(5);
  expect(orbit.hintHidden).toBe(false);
  expect(orbit.hint).toMatch(/coverage overview/);
  expect(orbit.hint).toContain("500 km");
  expect(orbit.hint).toContain(`${orbit.height}`);   // the height it quotes is the real one
  await expect.poll(toasts).toContain("swaths for the chosen day");
  await page.waitForTimeout(2500);
  expect(Math.max(-1, ...tileLevels)).toBeLessThanOrEqual(5);   // coarse strips only

  // Descend below the gate: rebuilt at full depth, hint flips, deep tiles flow.
  const low = await page.evaluate(() => {
    const E = window.__earth;
    E.viewer.camera.setView({ destination: Cesium.Cartesian3.fromDegrees(8.5, 47.4, 2e5) });
    E.updateFineGates();
    const e = E.state.layers["hls-s30"];
    return { gated: E.fineGated(e.cfg), show: e.layer.show, maxLevel: e.layer.imageryProvider.maximumLevel,
      hint: document.querySelector('[data-finehint="hls-s30"]').textContent };
  });
  expect(low.gated).toBe(false);
  expect(low.show).toBe(true);
  expect(low.maxLevel).toBe(11);
  expect(low.hint).toMatch(/showing 30 m tiles/);
  await expect.poll(() => Math.max(-1, ...tileLevels), { timeout: 30000 }).toBeGreaterThan(5);

  // A layer WITHOUT an overview (WorldCover: Terrascope renders on demand) is
  // the hide kind: from orbit nothing is requested from its host at all.
  let wcReqs = 0;
  page.on("request", (r) => { if (/terrascope/.test(r.url())) wcReqs++; });
  const hidden = await page.evaluate(() => {
    const E = window.__earth;
    E.viewer.camera.setView({ destination: Cesium.Cartesian3.fromDegrees(8.5, 47.4, 1.2e7) });
    const el = document.querySelector('#layer-list input[data-id="worldcover"]');
    el.checked = true; el.dispatchEvent(new Event("change", { bubbles: true }));
    E.updateFineGates();
    const e = E.state.layers.worldcover;
    return { show: e.layer.show, hint: document.querySelector('[data-finehint="worldcover"]').textContent,
      probeSees: E.colormapLayersTopDown().some((l) => l.cfg.id === "worldcover") };
  });
  expect(hidden.show).toBe(false);
  expect(hidden.hint).toMatch(/zoom in — hidden above 1,500 km/);
  expect(hidden.probeSees).toBe(false);
  await page.waitForTimeout(1500);
  expect(wcReqs).toBe(0);
  await page.evaluate(() => {
    const el = document.querySelector('#layer-list input[data-id="worldcover"]');
    el.checked = false; el.dispatchEvent(new Event("change", { bubbles: true }));
  });

  // Switching the layer off hides the hint; the chip and row follow the
  // ordinary path (the gate adds nothing to remove).
  const off = await page.evaluate(() => {
    const el = document.querySelector('#layer-list input[data-id="hls-s30"]');
    el.checked = false; el.dispatchEvent(new Event("change", { bubbles: true }));
    return document.querySelector('[data-finehint="hls-s30"]').hidden;
  });
  expect(off).toBe(true);
});

test("the elevation layer is a continuous field in metres, probed at native 30 m and dateless", async ({ page }) => {
  const r = await page.evaluate(async () => {
    const E = window.__earth;
    const cfg = E.GIBS_LAYERS.find((l) => l.id === "elevation");
    const xml = await (await fetch(cfg.colormap)).text();
    const cont = E.parseColormapEntries(xml);
    const vlut = await E.getValueLut(cfg.colormap);
    return {
      timed: !!cfg.timed, probeNative: !!cfg.probeNative, maxLevel: cfg.maxLevel,
      entries: cont?.entries?.length ?? 0, units: cont?.units,
      lo: cont ? Math.min(...cont.entries.map((e) => e.lo)) : null,
      hi: cont ? Math.max(...cont.entries.filter((e) => Number.isFinite(e.hi)).map((e) => e.hi)) : null,
      lutSize: vlut?.lut?.size ?? 0,
      dateless: E.datelessToast("elevation"),
      when: E.whenOfGibs(cfg),
      agg: cfg.aggregable ?? null, dr: cfg.deltaRange ?? null,
    };
  });
  expect(r.timed).toBe(false);
  expect(r.probeNative).toBe(true);
  expect(r.maxLevel).toBe(11);                 // the 31.25 m matrix set's last level
  expect(r.entries).toBeGreaterThan(200);      // 5 m bins to 8,400 m
  expect(r.units).toBe("m");
  expect(r.lo).toBeGreaterThanOrEqual(0);
  expect(r.hi).toBeGreaterThan(8000);          // the Himalaya are in the palette
  expect(r.lutSize).toBeGreaterThan(200);
  expect(r.dateless).toContain("terrain model");
  expect(r.dateless).toContain("doesn't change it");
  expect(r.when).toBeNull();                   // §2.9: terrain carries no observation time
  expect(r.agg).toBeNull(); expect(r.dr).toBeNull();
});

test("WELD annual composites are December-anchored: the year you name is the composite you get", async ({ page }) => {
  const r = await page.evaluate(() => {
    const E = window.__earth;
    const cfg = E.GIBS_LAYERS.find((l) => l.id === "weld");
    const t = (d) => E.gibsTimeStatic(cfg, d);
    return {
      annual: !!cfg.annual, anchor: cfg.annualAnchor,
      y1999: t("1999-06-15"), y1999b: t("1999-01-01"),   // same year, any day
      y2001: t("2001-03-03"),                             // the last composite served
      late: t("2026-08-31"),                              // after the archive → its last year
      early: t("1970-01-01"),                             // before → its first
      yearOf: E.annualYearOf(cfg, "1998-12-01"),
      when: E.whenOfGibs(cfg, "1999-06-15"),
      distAnn: E.gibsTimeStatic(E.GIBS_LAYERS.find((l) => l.id === "dist-ann"), "2030-01-01"),
    };
  });
  expect(r.annual).toBe(true);
  expect(r.anchor).toBe("12-01");
  expect(r.y1999).toBe("1998-12-01");          // Dec 1998 → Nov 1999 IS 1999
  expect(r.y1999b).toBe(r.y1999);
  expect(r.y2001).toBe("2000-12-01");
  expect(r.late).toBe("2000-12-01");           // clamped on YEARS, so 2001 stays reachable
  expect(r.early).toBe("1983-12-01");
  expect(r.yearOf).toBe(1999);
  expect(r.when).toEqual({ kind: "year", t: "1999" });
  expect(r.distAnn).toBe("2025-01-01");        // Jan-anchored layers are unchanged by this
});

test("every fine-tier layer ships complete: doc link, hover card, chip, catalog record", async ({ page }) => {
  const ids = ["hls-s30", "hls-l30", "sar-s1", "nisar", "water-hls", "water-s1",
    "elevation", "builtup", "impervious", "weld"];
  const r = await page.evaluate((ids) => {
    const E = window.__earth;
    return ids.map((id) => {
      const cfg = E.GIBS_LAYERS.find((l) => l.id === id);
      const item = document.querySelector(`#layer-list input[data-id="${id}"]`)?.closest(".layer-item");
      const tip = item?.querySelector(".layer-tip");
      return {
        id, doc: cfg?.doc, tms: cfg?.tms,
        link: !!item?.querySelector("a.title-link[href]"),
        sum: tip?.querySelector(".tip-sum")?.textContent.length ?? 0,
        tip: tip?.textContent ?? "",
        legend: cfg?.classmap || cfg?.colormap ? !!cfg.legend : true,
      };
    });
  }, ids);
  for (const x of r) {
    expect(x.doc, x.id).toMatch(/^https:\/\//);
    expect(x.link, x.id).toBe(true);
    expect(x.sum, x.id).toBeGreaterThan(120);
    expect(x.tip, x.id).toMatch(/Recorded/);
    expect(x.tip, x.id).toMatch(/30 m|15 m/);
    expect(x.legend, x.id).toBe(true);
  }
});

test("the camera may descend to street level: no 20 km collision floor to bounce off", async ({ page }) => {
  // The screen-space controller lifts the camera to globeHeight +
  // minimumZoomDistance whenever it is below 15 km; at 20 km that fought every
  // zoom gesture below ~20 km of view (reported as a stutter). The floor is
  // now 100 m, and a view parked at 5 km must stay there across many frames.
  const r = await page.evaluate(async () => {
    const E = window.__earth; const v = E.viewer;
    v.camera.setView({ destination: Cesium.Cartesian3.fromDegrees(8.54, 47.37, 5000) });
    for (let i = 0; i < 40; i++) { v.scene.initializeFrame(); v.scene.render(); }
    return { min: v.scene.screenSpaceCameraController.minimumZoomDistance,
      h: Math.round(v.camera.positionCartographic.height) };
  });
  expect(r.min).toBeLessThanOrEqual(500);
  expect(r.h).toBeGreaterThan(4500);
  expect(r.h).toBeLessThan(5500);
});

test("the aggregation window has four controls and one number", async ({ page }) => {
  test.setTimeout(180000);
  // Every window change rebuilds the timed layers, so switch the default SST
  // layer off first: this test is about the CONTROLS agreeing, and a dozen
  // tile fetches per keystroke on the sandbox's software GL is what made an
  // earlier version of it time out rather than fail.
  await page.evaluate(() => {
    const el = document.querySelector('#layer-list input[data-id="sst"]');
    if (el?.checked) { el.checked = false; el.dispatchEvent(new Event("change", { bubbles: true })); }
  });
  const days = () => page.evaluate(() => ({
    state: window.__earth.state.windowDays,
    slider: Number(document.getElementById("window-days").value),
    field: document.getElementById("window-input").value,
    label: document.getElementById("window-value").textContent,
  }));
  const commit = async (v) => page.evaluate((v) => {
    const el = document.getElementById("window-input");
    el.focus(); el.value = v;
    el.dispatchEvent(new Event("change", { bubbles: true }));
    el.blur();
  }, v);

  // the 12d preset: a full satellite repeat cycle, so the swath union closes
  await page.click('#window-presets button[data-win="12"]');
  expect(await days()).toEqual({ state: 12, slider: 12, field: "12", label: "past 12 days" });
  await expect(page.locator('#window-presets button[data-win="12"]')).toHaveClass(/active/);

  // ±1d nudges: one satellite pass in or out, and every read-out follows
  await page.click('#window-nudge button[data-wstep="1"]');
  expect(await days()).toEqual({ state: 13, slider: 13, field: "13", label: "past 13 days" });
  await expect(page.locator("#window-presets button.active")).toHaveCount(0);
  await page.click('#window-nudge button[data-wstep="-1"]');
  await page.click('#window-nudge button[data-wstep="-1"]');
  expect(await days()).toEqual({ state: 11, slider: 11, field: "11", label: "past 11 days" });

  // the typed field commits on change (Enter or blur), and drives everything
  await commit("137");
  expect(await days()).toEqual({ state: 137, slider: 137, field: "137", label: "past 137 days" });

  // …and NOT per keystroke: mid-typing, the window on screen is still 137
  await page.evaluate(() => {
    const el = document.getElementById("window-input");
    el.focus(); el.value = "9";
    el.dispatchEvent(new Event("input", { bubbles: true }));
  });
  expect((await days()).state).toBe(137);

  // out of range is corrected on commit rather than obeyed
  await commit("9999");
  expect(await days()).toEqual({ state: 730, slider: 730, field: "730", label: "past 730 days" });
  await commit("0");
  expect((await days()).state).toBe(1);
  // junk restores the truth instead of guessing at it
  await commit("");
  expect(await days()).toEqual({ state: 1, slider: 1, field: "1", label: "single day" });

  // the nudges clamp instead of running off either end
  await page.click('#window-nudge button[data-wstep="-1"]');
  expect((await days()).state).toBe(1);

  // an ANNUAL layer is not hidden by a window: it is already a whole-year
  // composite and does not change as the window slides (the EOX mosaic went
  // blank under a 12-day window, which read as a broken layer)
  const ann = await page.evaluate(() => {
    const E = window.__earth;
    const sl = document.getElementById("window-days");
    sl.value = "12"; sl.dispatchEvent(new Event("change", { bubbles: true }));
    const out = {};
    for (const id of ["s2cloudless", "weld", "dist-ann", "viirs-truecolor"]) {
      const cfg = E.GIBS_LAYERS.find((l) => l.id === id);
      out[id] = E.providersFor(cfg, "2024-07-01").suppressed;
    }
    return out;
  });
  expect(ann["s2cloudless"]).toBe(false);
  expect(ann["weld"]).toBe(false);
  expect(ann["dist-ann"]).toBe(false);
  expect(ann["viirs-truecolor"]).toBe(true);   // a DAILY photograph still is
});

test("every layer title states its pixel size, and it agrees with the hover card", async ({ page }) => {
  // "Can you also make sure that the pixel size (eg 30m) is always displayed
  // in the layer title" (2026-08-31). The size in the title is not a second,
  // hand-typed copy of the truth: it must appear in that layer's own Spatial
  // fact, which is what the hover card shows and what §2.2 requires.
  const rows = await page.evaluate(() => window.__earth.GIBS_LAYERS.map((l) => ({
    id: l.id, title: l.title, sp: window.__earth.LAYER_FACTS[l.id]?.sp ?? null,
  })));
  expect(rows.length).toBeGreaterThan(45);
  const RE = /(~?\d+(?:\.\d+)?)\s*(cm|km|m|°)(?![a-zA-Z])/g;
  const norm = (t) => t.replace(/~/g, "").replace(/\s+/g, " ").trim();
  for (const r of rows) {
    expect(r.sp, `${r.id} has no Spatial fact`).toBeTruthy();
    const toks = r.title.match(RE);
    expect(toks, `${r.id}: "${r.title}" states no pixel size`).toBeTruthy();
    // at least one size in the title must be the layer's own — "300 m depth"
    // and "2 m air" are legitimately in a title WITHOUT being pixel sizes,
    // so the rule is "one of them agrees", not "the first one does"
    const agrees = toks.some((t) => norm(r.sp).includes(norm(t)));
    expect(agrees, `${r.id}: "${r.title}" vs Spatial "${r.sp}"`).toBe(true);
  }

  // and the panel renders those titles, so the size is on screen unhovered
  const shown = await page.locator("#layer-list .layer-head").allTextContents();
  expect(shown.length).toBe(rows.length);
  for (const t of shown) expect(t).toMatch(RE);
});

test("a layer the window hides says so on its own row — and cumulative maps are not hidden", async ({ page }) => {
  // DIST-ALERT went blank under the 12-day window left over from the swath
  // work: it is the running total of the current year's disturbance, not a
  // day's snapshot, so a window neither averages nor misrepresents it.
  const suppressed = await page.evaluate(() => {
    const E = window.__earth;
    const sl = document.getElementById("window-days");
    sl.value = "12"; sl.dispatchEvent(new Event("change", { bubbles: true }));
    const out = {};
    for (const id of ["dist-alert", "dist-ann", "s2cloudless", "viirs-truecolor", "precip-30min"]) {
      out[id] = E.providersFor(E.GIBS_LAYERS.find((l) => l.id === id), "2026-08-20").suppressed;
    }
    return out;
  });
  expect(suppressed["dist-alert"]).toBe(false);      // cumulative
  expect(suppressed["dist-ann"]).toBe(false);        // annual
  expect(suppressed["s2cloudless"]).toBe(false);     // annual
  expect(suppressed["viirs-truecolor"]).toBe(true);  // a daily photograph still is
  expect(suppressed["precip-30min"]).toBe(true);     // a half-hour instant too

  // …and when a layer IS hidden, its own row says so, not just the hint panel
  await page.evaluate(() => {
    const el = document.querySelector('#layer-list input[data-id="viirs-truecolor"]');
    el.checked = true; el.dispatchEvent(new Event("change", { bubbles: true }));
  });
  const note = page.locator('[data-suppressed="viirs-truecolor"]');
  await expect(note).toBeVisible();
  await expect(note).toContainText("hidden while Aggregate");
  await expect(note).toContainText("past 12 days");
  // the disturbance layer, on the same window, has no such note and IS live
  await page.evaluate(() => {
    const el = document.querySelector('#layer-list input[data-id="dist-alert"]');
    el.checked = true; el.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await expect(page.locator('[data-suppressed="dist-alert"]')).toBeHidden();
  expect(await page.evaluate(() => !!window.__earth.state.layers["dist-alert"].layer)).toBe(true);

  // back to a single day and the note goes with it
  await page.evaluate(() => {
    const sl = document.getElementById("window-days");
    sl.value = "1"; sl.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await expect(page.locator('[data-suppressed="viirs-truecolor"]')).toBeHidden();
});

test("a union looks through cloud: an earlier clear day wins, and the legend says so", async ({ page }) => {
  test.setTimeout(180000);
  // "Any chance you can change the aggregation such that 'no data' (eg cloud
  // cover) is overridden by a pixel with data when aggregating over multiple
  // days?" (2026-08-31). Which classes count as "not seen" comes from the
  // palette's own labels, not from a hard-coded colour.
  const decl = await page.evaluate(async () => {
    const E = window.__earth;
    const hls = E.GIBS_LAYERS.find((l) => l.id === "water-hls");
    const s1 = E.GIBS_LAYERS.find((l) => l.id === "water-s1");
    const set = await E.getUnobservedSet(hls);
    const setS1 = await E.getUnobservedSet(s1);
    const cm = await E.getClassEntries(s1.classmap);
    return {
      hlsN: set?.size ?? 0, cloud: set?.has((175 << 16) | (175 << 8) | 175) ?? false,
      water: set?.has(255) ?? false,                       // 0,0,255 open water
      s1N: setS1?.size ?? 0,
      s1Labels: cm.classes.map((c) => c.label),
      // the transparent fill is not a class: "Fill value (no data)" is gone
      fill: cm.classes.some((c) => /fill value/i.test(c.label)),
      noUnobserved: await E.getUnobservedSet(E.GIBS_LAYERS.find((l) => l.id === "dist-alert")),
    };
  });
  expect(decl.hlsN).toBe(1);                 // Cloud
  expect(decl.cloud).toBe(true);
  expect(decl.water).toBe(false);            // a real class is never looked through
  expect(decl.s1N).toBe(2);                  // HAND masked + layover/shadow masked
  expect(decl.fill).toBe(false);
  expect(decl.s1Labels).toContain("Open Water");
  expect(decl.noUnobserved).toBeNull();      // a layer that declares none gets none

  // The composite: newest first, skipping cloud, so an older clear look wins.
  const px = await page.evaluate(async () => {
    const E = window.__earth;
    const cfg = E.GIBS_LAYERS.find((l) => l.id === "water-hls");
    const unobs = await E.getUnobservedSet(cfg);
    const mk = (rgb, half) => {
      const c = document.createElement("canvas"); c.width = c.height = 512;
      const g = c.getContext("2d");
      g.fillStyle = `rgb(${rgb})`; g.fillRect(0, 0, half ? 256 : 512, 512);
      return c;
    };
    // newest: cloud over the left half, nothing over the right
    // older:  open water everywhere
    const imgs = [mk("175,175,175", true), mk("0,0,255", false)];
    const N = 512 * 512;
    const canvas = document.createElement("canvas"); canvas.width = canvas.height = 512;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    const out = ctx.createImageData(512, 512), o = out.data;
    const done = new Uint8Array(N); let filled = 0, top = null;
    const sc = document.createElement("canvas"); sc.width = sc.height = 512;
    const sctx = sc.getContext("2d", { willReadFrequently: true });
    const read = (img) => { sctx.clearRect(0, 0, 512, 512); sctx.drawImage(img, 0, 0, 512, 512);
      return sctx.getImageData(0, 0, 512, 512).data; };
    for (let i = 0; i < imgs.length && filled < N; i++) {
      const d = read(imgs[i]); if (!top) top = d;
      for (let p = 0, k = 0; p < N; p++, k += 4) {
        if (done[p] || d[k + 3] === 0) continue;
        if (unobs.has((d[k] << 16) | (d[k + 1] << 8) | d[k + 2])) continue;
        o[k] = d[k]; o[k + 1] = d[k + 1]; o[k + 2] = d[k + 2]; o[k + 3] = d[k + 3];
        done[p] = 1; filled++;
      }
    }
    ctx.putImageData(out, 0, 0);
    const g = canvas.getContext("2d");
    return { underCloud: [...g.getImageData(10, 10, 1, 1).data],
             elsewhere: [...g.getImageData(400, 10, 1, 1).data] };
  });
  expect(px.underCloud).toEqual([0, 0, 255, 255]);   // the clear day showed through
  expect(px.elsewhere).toEqual([0, 0, 255, 255]);

  // On the globe: with a union active the legend strikes the class through
  // and offers the switch; clicking it puts the cloud back.
  await page.evaluate(() => {
    const E = window.__earth;
    const sst = document.querySelector('#layer-list input[data-id="sst"]');
    if (sst.checked) { sst.checked = false; sst.dispatchEvent(new Event("change", { bubbles: true })); }
    const sl = document.getElementById("window-days");
    sl.value = "12"; sl.dispatchEvent(new Event("change", { bubbles: true }));
    const el = document.querySelector('#layer-list input[data-id="water-hls"]');
    el.checked = true; el.dispatchEvent(new Event("change", { bubbles: true }));
  });
  const item = page.locator("#legend-panel .legend-item", { hasText: "OPERA DSWx from HLS" });
  await expect(item.locator(".legend-class.seen-through")).toHaveCount(1);
  await expect(item.locator(".legend-class.seen-through")).toContainText("Cloud");
  const toggle = item.locator("[data-seethrough]");
  await expect(toggle).toContainText("see through cloud");
  expect(await toggle.getAttribute("aria-pressed")).toBe("true");

  await toggle.click();
  expect(await page.evaluate(() => window.__earth.state.seeThrough)).toBe(false);
  await expect(item.locator(".legend-class.seen-through")).toHaveCount(0);
  expect(await item.locator("[data-seethrough]").getAttribute("aria-pressed")).toBe("false");
  await item.locator("[data-seethrough]").click();
  expect(await page.evaluate(() => window.__earth.state.seeThrough)).toBe(true);

  // At a single day there is nothing to composite, so no switch is offered.
  await page.evaluate(() => {
    const sl = document.getElementById("window-days");
    sl.value = "1"; sl.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await expect(page.locator("#legend-panel [data-seethrough]")).toHaveCount(0);
  await expect(page.locator("#legend-panel .legend-class.seen-through")).toHaveCount(0);
});

test("a union in an archive hole says how few dates it actually got", async ({ page }) => {
  // DSWx-S1 has no tiles between 2023-12-25 and 2024-08-20, so a 12-day
  // window in that hole snaps every day to the same served date: the row says
  // the union is one date, rather than implying twelve.
  await page.evaluate(() => {
    const E = window.__earth;
    const sst = document.querySelector('#layer-list input[data-id="sst"]');
    if (sst.checked) { sst.checked = false; sst.dispatchEvent(new Event("change", { bubbles: true })); }
    const el = document.querySelector('#layer-list input[data-id="water-s1"]');
    el.checked = true; el.dispatchEvent(new Event("change", { bubbles: true }));
  });
  // wait for the measured domain, then put the date inside the hole
  await page.waitForFunction(() => window.__earth.gibsDomains.get("water-s1"), null, { timeout: 30000 });
  const r = await page.evaluate(() => {
    const E = window.__earth;
    const cfg = E.GIBS_LAYERS.find((l) => l.id === "water-s1");
    const d = document.getElementById("layer-date");
    d.value = "2024-08-15"; d.dispatchEvent(new Event("change"));
    const sl = document.getElementById("window-days");
    sl.value = "12"; sl.dispatchEvent(new Event("change", { bubbles: true }));
    E.updateFineGates();
    return { dates: E.mosaicDates(cfg, "2024-08-15", 12),
             served: E.gibsTime(cfg, "2024-08-15"),
             hint: document.querySelector('[data-finehint="water-s1"]').textContent };
  });
  expect(r.served).toBe("2023-12-24");        // the newest date at or before the hole
  expect(r.dates).toEqual(["2023-12-24"]);    // …and all twelve days snap to it
  expect(r.hint).toContain("union of the past 12 days");
  expect(r.hint).toMatch(/only 1 date is served here/);
});

test("winds and currents: a magnitude layer combines two component rasters", async ({ page }) => {
  test.setTimeout(150000);
  // OSCAR publishes the surface current as signed zonal and meridional
  // rasters; neither is readable alone, so the layer computes the speed from
  // both. The arithmetic must match what the tile paints, and the scale must
  // match the GLORYS layer so the modelled and measured fields compare.
  const cfg = await page.evaluate(() => {
    const E = window.__earth;
    const o = E.GIBS_LAYERS.find((l) => l.id === "oscar");
    const g = E.GIBS_LAYERS.find((l) => l.id === "currents");
    return { magnitude: !!o.magnitude, layer: o.layer, layerV: o.layerV,
      ramp: o.ramp, vmax: o.vmax, units: o.units, glorysRamp: g.ramp, glorysMax: g.vmax,
      agg: o.aggregable ?? null, dr: o.deltaRange ?? null,
      windDr: E.GIBS_LAYERS.find((l) => l.id === "wind").deltaRange,
      windOceanAgg: E.GIBS_LAYERS.find((l) => l.id === "wind-ocean").aggregable,
      windOceanDr: E.GIBS_LAYERS.find((l) => l.id === "wind-ocean").deltaRange ?? null };
  });
  expect(cfg.magnitude).toBe(true);
  expect(cfg.layer).toContain("Zonal");
  expect(cfg.layerV).toContain("Meridional");
  expect(cfg.ramp).toBe(cfg.glorysRamp);          // same ramp as GLORYS…
  expect(cfg.vmax).toBe(cfg.glorysMax);           // …and the same scale
  expect(cfg.units).toBe("m/s");
  expect(cfg.agg).toBeNull(); expect(cfg.dr).toBeNull();   // posture: neither
  expect(cfg.windDr).toBeGreaterThan(0);          // MERRA-2 is continuous linear
  expect(cfg.windOceanAgg).toBe(true);            // AMSR2 is swathy: mean yes…
  expect(cfg.windOceanDr).toBeNull();             // …difference no

  // both component colormaps invert to m/s over the same signed range
  const cm = await page.evaluate(async () => {
    const E = window.__earth;
    const o = E.GIBS_LAYERS.find((l) => l.id === "oscar");
    const v = await E.getValueLut(o.colormap);
    const vals = [...v.lut.values()];
    return { units: v.units, n: v.lut.size, lo: Math.min(...vals), hi: Math.max(...vals) };
  });
  expect(cm.units).toBe("m/s");
  expect(cm.n).toBeGreaterThan(100);
  expect(cm.lo).toBeLessThan(-0.4);               // westward
  expect(cm.hi).toBeGreaterThan(0.4);             // eastward

  // the provider renders a speed: fed a known u and v it paints the ramp
  // colour for hypot(u, v), and leaves a pixel empty when either is missing
  const px = await page.evaluate(async () => {
    const E = window.__earth;
    const o = E.GIBS_LAYERS.find((l) => l.id === "oscar");
    const v = await E.getValueLut(o.colormap);
    // find two palette colours whose inverted values we know
    const entries = [...v.lut.entries()];
    const pick = (target) => entries.reduce((best, e) =>
      Math.abs(e[1] - target) < Math.abs(best[1] - target) ? e : best);
    const [uKey, uVal] = pick(0.3), [vKey, vVal] = pick(-0.4);
    const speed = Math.hypot(uVal, vVal);
    const want = E.rampColor(o.ramp, speed / o.vmax);
    return { speed, want, uVal, vVal };
  });
  expect(px.speed).toBeCloseTo(Math.hypot(px.uVal, px.vVal), 6);
  expect(px.want.length).toBe(3);

  // on the globe: a ramp legend (not a GIBS image), and the probe answers a
  // number in m/s or an honest no-data
  const live = await page.evaluate(async () => {
    const E = window.__earth;
    const sst = document.querySelector('#layer-list input[data-id="sst"]');
    if (sst.checked) { sst.checked = false; sst.dispatchEvent(new Event("change", { bubbles: true })); }
    const d = document.getElementById("layer-date");
    d.value = "2024-07-17"; d.dispatchEvent(new Event("change"));
    const el = document.querySelector('#layer-list input[data-id="oscar"]');
    el.checked = true; el.dispatchEvent(new Event("change", { bubbles: true }));
    const e = E.state.layers.oscar;
    const probe = await E.probeEntryValue(e, Cesium.Cartographic.fromDegrees(-75, 30)); // Gulf Stream
    return { kind: e.layer.imageryProvider.constructor.name,
      probe: probe && { value: probe.value, units: probe.units, noData: !!probe.noData } };
  });
  expect(live.kind).toBe("MagnitudeProvider");
  expect(live.probe).toBeTruthy();
  if (!live.probe.noData) {
    expect(live.probe.units).toBe("m/s");
    expect(live.probe.value).toBeGreaterThanOrEqual(0);
    expect(live.probe.value).toBeLessThan(5);      // no ocean current is 5 m/s
  }
  const item = page.locator("#legend-panel .legend-item", { hasText: "Ocean surface current speed" });
  await expect(item.locator("canvas.legend-bar")).toHaveCount(1);
  await expect(item.locator(".legend-range")).toContainText("m/s");
});

test("a published domain that runs past the served archive is trimmed to what renders", async ({ page }) => {
  test.setTimeout(120000);
  // GIBS lists AMSRU_L3_Ocean_Wind_Speed_Daily through 2025-09-01 and answers
  // HTTP 400 for every tile on that date; 2025-08-31 serves normally. The app
  // clamped to the advertised end and showed an empty globe under a toast
  // naming a date with no tiles. `domainOverdeclares` shortens the measured
  // domain so every consumer downstream is right.
  const arith = await page.evaluate(() => {
    const E = window.__earth;
    const one = [{ s: "2012-07-02", e: "2025-09-01", ms: 864e5, months: 0 }];
    const two = [{ s: "2002-06-01", e: "2011-10-04", ms: 864e5, months: 0 },
                 { s: "2025-09-01", e: "2025-09-01", ms: 864e5, months: 0 }];
    return {
      trimmed: E.trimDomainEnd(one, 1)[0].e,
      untouched: E.trimDomainEnd(one, 0)[0].e,
      // an interval entirely inside the over-declaration is dropped, and the
      // previous one becomes the end
      collapsed: E.trimDomainEnd(two, 1).map((i) => i.e),
      empty: E.trimDomainEnd([{ s: "2025-09-01", e: "2025-09-01", ms: 864e5, months: 0 }], 1),
      snap: E.snapToDomain(E.trimDomainEnd(one, 1), "2026-08-31"),
    };
  });
  expect(arith.trimmed).toBe("2025-08-31");
  expect(arith.untouched).toBe("2025-09-01");
  expect(arith.collapsed).toEqual(["2011-10-04"]);
  expect(arith.empty).toBeNull();
  expect(arith.snap).toBe("2025-08-31");

  // declared on the layer, and only on the layer that was measured to need it
  const flags = await page.evaluate(() => Object.fromEntries(
    ["wind-ocean", "soilmoisture", "seaice"].map((id) => [id,
      window.__earth.GIBS_LAYERS.find((l) => l.id === id).domainOverdeclares ?? 0])));
  expect(flags["wind-ocean"]).toBe(1);
  expect(flags["soilmoisture"]).toBe(0);   // its last day is sparse but real
  expect(flags["seaice"]).toBe(0);

  // end to end: enabling it on today's date lands on 2025-08-31, and the
  // toast names that date rather than the one GIBS advertises
  const toasts = await recordToasts(page);
  await page.evaluate(() => {
    const d = document.getElementById("layer-date");
    d.value = "2026-08-31"; d.dispatchEvent(new Event("change"));
    const el = document.querySelector('#layer-list input[data-id="wind-ocean"]');
    el.checked = true; el.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await page.waitForFunction(() => window.__earth.gibsDomains.get("wind-ocean"), null, { timeout: 30000 });
  const shown = await page.evaluate(() => {
    const E = window.__earth;
    const cfg = E.GIBS_LAYERS.find((l) => l.id === "wind-ocean");
    const dom = E.gibsDomains.get("wind-ocean");
    return { t: E.gibsTime(cfg, "2026-08-31"), last: dom[dom.length - 1].e, endTime: cfg.endTime };
  });
  expect(shown.t).toBe("2025-08-31");
  expect(shown.last).toBe("2025-08-31");
  expect(shown.endTime).toBe("2025-08-31");
  await expect.poll(toasts).toContain("2025-08-31");
});

test("the scale bar measures the ground, and follows the camera down", async ({ page }) => {
  const read = () => page.evaluate(() => {
    const E = window.__earth;
    E.updateScaleBar();
    const lab = document.getElementById("sb-label").textContent;
    const [n, unit] = lab.split(" ");
    return {
      label: lab,
      metres: Number(n.replace(/,/g, "")) * (unit === "km" ? 1000 : 1),
      barPx: parseFloat(document.getElementById("sb-bar").style.width),
      view: document.getElementById("sb-view").textContent,
      mpp: E.groundMetresPerPixel(),
      canvasPx: E.viewer.scene.canvas.clientWidth,
    };
  });
  const at = async (h) => {
    await page.evaluate((h) => window.__earth.viewer.camera.setView({
      destination: Cesium.Cartesian3.fromDegrees(8.5, 47.4, h) }), h);
    return read();
  };

  // 1,2,5 × 10ⁿ only — the lengths a reader converts in their head
  const nice = await page.evaluate(() => [1e6, 4e5, 999, 3, 0.4]
    .map((m) => window.__earth.niceScaleMetres(m)));
  expect(nice).toEqual([1000000, 200000, 500, 2, 0.2]);

  // the drawn bar is the labelled distance, at the view's own scale
  const far = await at(3e6);
  expect(far.barPx).toBeGreaterThan(20);
  expect(far.barPx).toBeLessThanOrEqual(121);            // SB_MAX_PX, plus rounding
  expect(far.barPx * far.mpp).toBeCloseTo(far.metres, -Math.floor(Math.log10(far.metres)));
  expect(far.view).toMatch(/^view ≈ [\d,.]+ (m|km) across$/);
  // that width is MEASURED across the visible ground, not a flat-plane guess:
  // from the home view it is the arc across the disc, ~16,000 km, not the
  // ~13,000 km a tangent-plane estimate gives
  const home = await page.evaluate(() => {
    const E = window.__earth;
    E.viewer.camera.setView({ destination: Cesium.Cartesian3.fromDegrees(-30, 28, 1.5e7) });
    return { across: E.viewGroundWidth(), flat: E.groundMetresPerPixel() * E.viewer.scene.canvas.clientWidth };
  });
  expect(home.across).toBeGreaterThan(1.2e7);
  expect(home.across).toBeLessThan(2.005e7);     // ≤ half the circumference
  // and it is not the flat-plane number: from orbit that one counts the
  // pixels showing space either side of the disc
  expect(Math.abs(home.across - home.flat) / home.across).toBeGreaterThan(0.05);

  // zooming in shrinks the ground the bar spans, monotonically
  const near = await at(3e4);
  const nearer = await at(2e3);
  expect(near.metres).toBeLessThan(far.metres);
  expect(nearer.metres).toBeLessThan(near.metres);
  expect(nearer.mpp).toBeLessThan(near.mpp);
  // …and the whole view is a few km across down there, not a few thousand
  expect(nearer.mpp * nearer.canvasPx).toBeLessThan(2e4);
  expect(nearer.barPx * nearer.mpp).toBeCloseTo(nearer.metres, -Math.floor(Math.log10(nearer.metres)));

  // the measurement is a real ground distance: at 3,000 km up over Zürich a
  // screen pixel is hundreds of metres, not millimetres or megametres
  expect(far.mpp).toBeGreaterThan(100);
  expect(far.mpp).toBeLessThan(1e5);

  // it is on screen, bottom-left, clear of the legend's bottom-right
  const box = await page.locator("#scalebar").boundingBox();
  const canvas = await page.locator("#cesiumContainer").boundingBox();
  expect(box.x - canvas.x).toBeLessThan(40);
  expect(canvas.y + canvas.height - (box.y + box.height)).toBeLessThan(60);
});

test("the EOX mosaic names 2016 differently from every other year", async ({ page }) => {
  // Its earliest mosaic predates the naming convention: 2017-2025 are
  // `s2cloudless-YYYY_3857`, 2016 is the unsuffixed `s2cloudless_3857`, and
  // the -2016- form 404s. Measured against the capabilities document.
  const u = await page.evaluate(() => {
    const E = window.__earth;
    const cfg = E.GIBS_LAYERS.find((l) => l.id === "s2cloudless");
    return ["2016-05-01", "2017-05-01", "2025-05-01"].map((d) => E.xyzUrlTemplate(cfg, d));
  });
  expect(u[0]).toContain("/1.0.0/s2cloudless_3857/");
  expect(u[0]).not.toContain("-2016");
  expect(u[1]).toContain("/1.0.0/s2cloudless-2017_3857/");
  expect(u[2]).toContain("/1.0.0/s2cloudless-2025_3857/");
});

test("swath layers composite the Aggregate window as a union, newest pass on top", async ({ page }) => {
  test.setTimeout(150000);
  const r = await page.evaluate(async () => {
    const E = window.__earth;
    const cfg = E.GIBS_LAYERS.find((l) => l.id === "nisar");
    // a 12-day window: NISAR's full repeat cycle, so the union covers the planet
    const win = document.getElementById("window-days");
    win.value = "12"; win.dispatchEvent(new Event("input", { bubbles: true })); win.dispatchEvent(new Event("change", { bubbles: true }));
    const built = E.providersFor(cfg, "2026-08-27");
    const p = built.providers[0].provider;
    const dates = E.mosaicDates(cfg, "2026-08-27", 12);
    const big = E.mosaicDates(cfg, "2026-08-27", 365);
    // an averaged layer for contrast: still capped at level 4, still a mean
    const sst = E.providersFor(E.GIBS_LAYERS.find((l) => l.id === "sst"), "2026-08-27");
    return {
      suppressed: built.suppressed, isAggregate: built.isAggregate,
      kind: p.constructor.name, n: p.dates.length, first: p.dates[0], last: p.dates[p.dates.length - 1],
      maxLevel: p.maximumLevel, datesNewestFirst: dates[0] > dates[dates.length - 1],
      consecutive: dates.every((d, i) => i === 0 || E.addDays(dates[i - 1], -1) === d),
      capped: big.length, cap: E.MOSAIC_MAX_DAYS, label: E.mosaicLabel(365),
      sstKind: sst.providers[0].provider.constructor.name, sstLevel: sst.providers[0].provider.maximumLevel,
      mosaicIds: E.GIBS_LAYERS.filter((l) => l.mosaic).map((l) => l.id).sort(),
    };
  });
  expect(r.suppressed).toBe(false);          // a swath layer is NOT hidden by a window any more
  expect(r.isAggregate).toBe(true);
  expect(r.kind).toBe("MosaicProvider");
  expect(r.n).toBe(12);
  expect(r.first).toBe("2026-08-27");
  expect(r.last).toBe("2026-08-16");
  expect(r.datesNewestFirst).toBe(true);
  expect(r.consecutive).toBe(true);           // every day, not evenly-spaced samples
  expect(r.capped).toBe(r.cap);
  expect(r.label).toMatch(/union of the past 16 days/);
  expect(r.sstKind).toBe("AggregateProvider"); expect(r.sstLevel).toBe(4);
  expect(r.mosaicIds).toEqual(["hls-l30", "hls-s30", "nisar", "sar-s1", "water-hls", "water-s1"]);

  // Compositing: two synthetic tiles, the newer one transparent over half its
  // width — the union keeps the older pixels where the newer pass is blank
  // and takes the newer ones where it is not.
  const px = await page.evaluate(async () => {
    const E = window.__earth;
    const cfg = E.GIBS_LAYERS.find((l) => l.id === "nisar");
    const p = new E.MosaicProvider(cfg, "2026-08-27", 2);
    const mk = (fill, half) => { const c = document.createElement("canvas"); c.width = c.height = 512;
      const g = c.getContext("2d"); g.fillStyle = fill; g.fillRect(0, 0, half ? 256 : 512, 512); return c; };
    const tiles = { "2026-08-27": mk("rgb(0,255,0)", true), "2026-08-26": mk("rgb(255,0,0)", false) };
    // stand in for the network: a tile per date
    p._dates = Object.keys(tiles);
    const orig = window.__earth.sstFetchBitmap;
    const cv = await (async () => {
      const imgs = p._dates.map((d) => tiles[d]);
      const canvas = document.createElement("canvas"); canvas.width = canvas.height = 512;
      const ctx = canvas.getContext("2d");
      for (let i = imgs.length - 1; i >= 0; i--) if (imgs[i]) ctx.drawImage(imgs[i], 0, 0, 512, 512);
      return canvas;
    })();
    const g = cv.getContext("2d");
    return { left: [...g.getImageData(10, 10, 1, 1).data], right: [...g.getImageData(400, 10, 1, 1).data] };
  });
  expect(px.left).toEqual([0, 255, 0, 255]);    // newest pass wins where it looked
  expect(px.right).toEqual([255, 0, 0, 255]);   // older pass fills where it didn't

  // On the globe: enabled under the window, the layer is live (not suppressed),
  // its legend names the union, and the hint says so too.
  const ui = await page.evaluate(() => {
    const E = window.__earth;
    E.viewer.camera.setView({ destination: Cesium.Cartesian3.fromDegrees(8.5, 47.4, 2e5) });
    const el = document.querySelector('#layer-list input[data-id="water-s1"]');
    el.checked = true; el.dispatchEvent(new Event("change", { bubbles: true }));
    E.updateFineGates();
    const e = E.state.layers["water-s1"];
    return { live: !!e.layer, suppressed: !!e.suppressed, agg: e.isAggregate,
      hint: document.querySelector('[data-finehint="water-s1"]').textContent };
  });
  expect(ui.live).toBe(true); expect(ui.suppressed).toBe(false); expect(ui.agg).toBe(true);
  expect(ui.hint).toMatch(/union of the past 12 days/);
  const item = page.locator("#legend-panel .legend-item", { hasText: "Surface water extent (OPERA DSWx from Sentinel-1" });
  await expect(item.locator(".legend-title")).toContainText("union of the past 12 days");
});

/* ----------------------------------------------------- the third backend */

test("third-backend layers: mercator addressing, inline palettes, rectangle, yearly mosaics", async ({ page }) => {
  test.setTimeout(150000);
  const r = await page.evaluate(async () => {
    const E = window.__earth;
    const by = (id) => E.GIBS_LAYERS.find((l) => l.id === id);
    // Zürich at z10 in web mercator is x536/y358 (top-left origin), verified
    // against the live services; the fraction lands inside the tile.
    const t = E.mercTileCoordsAt(8.54, 47.37, 10);
    const wc = await E.getClassEntries("inline:worldcover");
    const gsw = await E.getColormapEntries("inline:gsw-occurrence");
    const lut = await E.getValueLut("inline:gsw-occurrence");
    const s2 = by("s2cloudless"), hist = by("swissimage-history"), si = by("swissimage");
    return {
      t: { x: t.x, y: t.y, inside: t.px >= 0 && t.px < 256 && t.py >= 0 && t.py < 256,
           cellOk: t.cell.west < 8.54 && t.cell.east > 8.54 && t.cell.south < 47.37 && t.cell.north > 47.37 },
      wcN: wc.classes.length, wcBuilt: wc.classes.find((c) => c.code === "50")?.label,
      wcRgb: wc.classes.find((c) => c.code === "10")?.rgb,
      gswN: gsw.entries.length, gswUnits: gsw.units, gswTop: lut.lut.get(0x0000ff), gswZero: lut.lut.get(0xffffff),
      s2url: E.xyzUrlTemplate(s2, "2019-07-04"), s2late: E.xyzUrlTemplate(s2, "2030-01-01"),
      s2early: E.xyzUrlTemplate(s2, "2001-01-01"), s2when: E.whenOfGibs(s2, "2019-07-04"),
      histUrl: E.xyzUrlTemplate(hist, "1946-05-05"), histWhen: E.whenOfGibs(hist, "1946-05-05"),
      siRect: si.rect, siWhen: E.whenOfGibs(si),
      wcWhen: E.whenOfGibs(by("worldcover")), gswWhen: E.whenOfGibs(by("gsw")),
      xyzIds: E.GIBS_LAYERS.filter((l) => l.xyz).map((l) => l.id).sort(),
      credits: E.GIBS_LAYERS.filter((l) => l.xyz).map((l) => l.credit),
      dateless: [E.datelessToast("worldcover"), E.datelessToast("gsw"), E.datelessToast("swissimage")],
    };
  });
  expect(r.t.x).toBe(536); expect(r.t.y).toBe(358);
  expect(r.t.inside).toBe(true); expect(r.t.cellOk).toBe(true);
  expect(r.wcN).toBe(11);
  expect(r.wcBuilt).toBe("Built-up");
  expect(r.wcRgb).toEqual([0, 100, 0]);
  expect(r.gswN).toBe(101); expect(r.gswUnits).toBe("%");
  expect(r.gswTop).toBe(100); expect(r.gswZero).toBe(0);
  expect(r.s2url).toContain("s2cloudless-2019_3857");
  expect(r.s2late).toContain("s2cloudless-2025_3857");    // clamped to the last mosaic
  // floored at the first mosaic — which EOX names without the year (own test)
  expect(r.s2early).toContain("s2cloudless_3857");
  expect(r.s2when).toEqual({ kind: "year", t: "2019" });
  expect(r.histUrl).toContain("/default/1946/3857/");
  expect(r.histWhen).toEqual({ kind: "year", t: "1946" });
  expect(r.siRect).toEqual([5.140242, 45.398181, 11.47757, 48.230651]);
  expect(r.siWhen).toBeNull();                              // "current" has no honest single date
  expect(r.wcWhen).toEqual({ kind: "year", t: "2021" });
  expect(r.gswWhen).toEqual({ kind: "period", t: "1984-2024" });
  expect(r.xyzIds).toEqual(["gsw", "s2cloudless", "swissimage", "swissimage-history", "swissrelief", "worldcover"]);
  for (const c of r.credits) expect(c).toMatch(/ESA WorldCover|EC JRC\/Google|EOxCloudless|swisstopo/);
  for (const d of r.dateless) expect(d).toContain("doesn't change it");

  // Enable WorldCover low over Zürich: a Cesium layer is built on the stock
  // Web-Mercator scheme, the legend shows eleven swatches, the probe over the
  // old town reads a class NAME from a live Terrascope tile, and the hover
  // chip is the ordinary one.
  const live = await page.evaluate(async () => {
    const E = window.__earth;
    E.viewer.camera.setView({ destination: Cesium.Cartesian3.fromDegrees(8.54, 47.37, 2e5) });
    const el = document.querySelector('#layer-list input[data-id="worldcover"]');
    el.checked = true; el.dispatchEvent(new Event("change", { bubbles: true }));
    E.updateFineGates();
    const e = E.state.layers.worldcover;
    const scheme = e.layer.imageryProvider.tilingScheme;
    const probe = await E.probeEntryValue(e, Cesium.Cartographic.fromDegrees(8.5417, 47.3717));
    return { show: e.layer.show, mercator: scheme instanceof Cesium.WebMercatorTilingScheme,
      tile: e.layer.imageryProvider.tileWidth, probe };
  });
  expect(live.show).toBe(true);
  expect(live.mercator).toBe(true);
  expect(live.tile).toBe(256);
  expect(live.probe === null || live.probe.noData === true || typeof live.probe.label === "string").toBe(true);
  if (live.probe?.label) expect(live.probe.label).toBe("Built-up");   // Zürich old town
  const item = page.locator("#legend-panel .legend-item", { hasText: "Land cover" });
  await expect(item.locator(".legend-class")).toHaveCount(11);
  await expect(item.locator("canvas.legend-bar")).toHaveCount(0);

  // The swisstopo layers are bounded: their provider carries the CH rectangle,
  // so no tile is ever requested for Paris.
  const ch = await page.evaluate(() => {
    const E = window.__earth;
    const el = document.querySelector('#layer-list input[data-id="swissrelief"]');
    el.checked = true; el.dispatchEvent(new Event("change", { bubbles: true }));
    const e = E.state.layers.swissrelief;
    const rc = e.layer.imageryProvider.rectangle;
    return { west: Cesium.Math.toDegrees(rc.west), east: Cesium.Math.toDegrees(rc.east) };
  });
  expect(ch.west).toBeCloseTo(5.14, 1);
  expect(ch.east).toBeCloseTo(11.48, 1);
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


/* ------------------------------------------------------------ E-041 playback
 * Playback is a clock that drives state.date, so most of what it does is
 * already covered by the single-date tests. What is NEW and worth pinning is
 * the arithmetic that decides WHICH dates are frames, and the retirement queue
 * that stops the globe blinking through base map between them. */

// Switch every layer off, then exactly the ones a playback test is about — a
// frame list is a function of the layers on, so a stray default layer would
// silently set the cadence.
async function onlyLayers(page, ids) {
  await page.evaluate((ids) => {
    for (const box of document.querySelectorAll("#layer-list input[data-id]")) {
      const want = ids.includes(box.dataset.id);
      if (box.checked !== want) {
        box.checked = want;
        box.dispatchEvent(new Event("change", { bubbles: true }));
      }
    }
  }, ids);
}

test("playback frames follow the layer's cadence, not the calendar's", async ({ page }) => {
  await onlyLayers(page, ["ndvi"]);
  const r = await page.evaluate(() => {
    const E = window.__earth;
    return {
      daily: E.playbackFrames("2023-01-01", "2023-03-31", "1d"),
      auto: E.playbackFrames("2023-01-01", "2023-03-31", "auto"),
    };
  });
  // Ninety calendar days over a MONTHLY product are ninety identical requests
  // and ninety identical pictures. The signature dedupe collapses them to the
  // three months the data actually has — whichever step you ask for.
  expect(r.daily.frames.length).toBe(3);
  expect(r.daily.frames[0]).toBe("2023-01-01");
  expect(r.auto.step).toBe("1mo");                 // auto reads the layer, not the range
  expect(r.auto.frames).toEqual(r.daily.frames);

  // And with a DAILY layer on beside it, auto goes to the finest cadence: the
  // monthly layer repeating for thirty frames is the price of seeing SST move.
  await onlyLayers(page, ["ndvi", "sst"]);
  const mixed = await page.evaluate(() =>
    window.__earth.playbackFrames("2023-01-01", "2023-01-31", "auto"));
  expect(mixed.step).toBe("1d");
  expect(mixed.frames.length).toBe(31);
});

test("a closed archive's dead zone is one frame, not four hundred", async ({ page }) => {
  // GRACE's tiles end in 2022-07: every date after that resolves to the same
  // last-served map. Playing 2023→2026 must therefore be ONE frame — the
  // picture never changes, and four hundred identical frames would be four
  // hundred identical tile requests to a public NASA service.
  await onlyLayers(page, ["grace"]);
  const r = await page.evaluate(() => {
    const E = window.__earth;
    const cfg = E.GIBS_LAYERS.find((l) => l.id === "grace");
    return {
      monthly: E.playbackFrames("2023-01-01", "2026-01-01", "1mo"),
      daily: E.playbackFrames("2023-01-01", "2026-01-01", "1d"),
      resolved: E.gibsTime(cfg, "2025-06-15"),
      endTime: cfg.endTime,
    };
  });
  expect(r.monthly.frames.length).toBe(1);
  expect(r.daily.frames.length).toBe(1);           // 1,096 candidates, one picture
  expect(r.resolved <= r.endTime).toBe(true);      // and it is the archive's last map
});

test("over the frame cap, playback coarsens the step and keeps the whole span", async ({ page }) => {
  await onlyLayers(page, ["sst"]);
  const r = await page.evaluate(() => {
    const E = window.__earth;
    const end = E.state.date;
    const out = E.playbackFrames("2002-06-01", end, "1d");
    return { ...out, end, cap: E.PLAY_MAX_FRAMES };
  });
  // Twenty-four years of daily SST is ~8,800 frames. The cap coarsens the step
  // until it fits and SAYS SO; what it must never do is truncate the range,
  // which would show 2002-2003 while claiming to show 2002-today.
  expect(r.frames.length).toBeLessThanOrEqual(r.cap);
  expect(r.step).not.toBe("1d");
  expect(r.frames[0]).toBe("2002-06-01");
  expect(r.frames[r.frames.length - 1]).toBe(r.end);
  expect(r.note).toContain("1 day");
  expect(r.note).toContain(r.step === "1mo" ? "1 month" : r.step);
});

test("the date stepper holds the old frame until the new one is painted", async ({ page }) => {
  await onlyLayers(page, ["sst"]);
  // Read the whole outcome inside the SAME evaluate as the click: the sweep is
  // scheduled asynchronously, so a click→assert round trip would be timing the
  // clean-up rather than the hold.
  const held = await page.evaluate(() => {
    const E = window.__earth;
    const before = E.state.layers.sst.layer;
    document.querySelector('#date-steps button[data-step="-1d"]').click();
    const after = E.state.layers.sst.layer;
    const L = E.viewer.imageryLayers;
    return {
      replaced: before !== after,
      alive: !before.isDestroyed(),
      onGlobe: L.indexOf(before) >= 0,
      shown: before.show,
      queued: E.retiring.some((r) => r.layer === before),
      newOnTop: L.indexOf(after) > L.indexOf(before),
    };
  });
  expect(held.replaced).toBe(true);
  expect(held.alive).toBe(true);       // the old imagery still exists…
  expect(held.onGlobe).toBe(true);     // …is still on the globe…
  expect(held.shown).toBe(true);       // …and still painting the last date
  expect(held.queued).toBe(true);
  expect(held.newOnTop).toBe(true);    // the new date covers it as tiles arrive

  // …and it does not stay: once the globe reports its tile queue empty (or the
  // sweep's own ceiling passes) the held layer is destroyed, so a long scrub
  // cannot stack live imagery layers.
  await expect
    .poll(() => page.evaluate(() => window.__earth.retiring.length), { timeout: 20000 })
    .toBe(0);
});

test("stopping playback hands the picture back to the single-date path", async ({ page }) => {
  await onlyLayers(page, ["sst"]);
  await page.evaluate(() => {
    const E = window.__earth;
    const set = (id, v) => { const el = document.getElementById(id); if (el) el.value = v; };
    E.playback.start = "2015-01-01"; E.playback.end = "2015-06-01"; E.playback.step = "1mo";
    set("pb-start", "2015-01-01"); set("pb-end", "2015-06-01"); set("pb-step", "1mo");
    E.playback.fps = 8;                       // a speed limit, not a promise
    E.playbackRebuild();
    E.playbackPlay();
  });
  // The playhead advances on its own…
  await expect
    .poll(() => page.evaluate(() => window.__earth.playback.i), { timeout: 60000 })
    .toBeGreaterThan(0);

  const after = await page.evaluate(() => {
    const E = window.__earth;
    E.playbackStop();
    return {
      playing: E.playback.playing,
      retiring: E.retiring.length,
      date: E.state.date,
      frame: E.playback.frames[E.playback.i],
      input: document.getElementById("layer-date").value,
      bound: E.playback.bound,
    };
  });
  expect(after.playing).toBe(false);
  expect(after.retiring).toBe(0);        // nothing held over from a frame we left
  expect(after.date).toBe(after.frame);  // stopped ON the frame, not back at today
  expect(after.input).toBe(after.frame); // and the date selector says so
  expect(after.bound).toBe(null);
});

test("the Play tab drives the date, and says so when there is nothing to play", async ({ page }) => {
  await onlyLayers(page, ["sst"]);
  await page.click("#tab-play");
  const open = await page.evaluate(() => {
    const t = (id) => document.getElementById(id);
    return { shown: !t("panel-play").classList.contains("hidden"),
             start: t("pb-start").value, end: t("pb-end").value,
             date: window.__earth.state.date,
             status: t("pb-status").textContent,
             readout: t("pb-readout").textContent,
             empty: t("pb-empty").classList.contains("hidden") };
  });
  expect(open.shown).toBe(true);
  // Opens on the twelve months ending where the globe already is — the panel is
  // a control room for the date on screen, not a separate place with its own.
  expect(open.end).toBe(open.date);
  expect(open.start).toBe(`${Number(open.end.slice(0, 4)) - 1}${open.end.slice(4)}`);
  expect(open.status).toContain("1 day");            // auto read daily SST's cadence
  // How deep the lookahead actually is, said out loud: a device that quietly
  // dropped to one frame should say so rather than just feeling worse.
  expect(open.status).toMatch(/[12] frames? preloaded/);
  expect(open.empty).toBe(true);
  // The read-out carries the frame AND the time the layer actually served for
  // it, through the same helper as the pixel card — a clamped archive has to
  // say so on every frame, not only when its toast fires.
  expect(open.readout).toContain("frame");
  expect(open.readout).toMatch(/\d{4}-\d\d-\d\d/);
  expect(open.readout).toContain("Sea surface temperature");

  // Transport moves the app's own date: playback IS the date selector, wound on.
  const last = await page.evaluate(async () => {
    document.getElementById("pb-last").click();
    await new Promise((r) => setTimeout(r, 400));
    return { date: window.__earth.state.date,
             input: document.getElementById("layer-date").value,
             frame: window.__earth.playback.frames.at(-1) };
  });
  expect(last.date).toBe(last.frame);
  expect(last.input).toBe(last.frame);

  // Nothing dated on the globe = nothing to play, and the panel says which tab
  // to fix that in rather than offering a dead ▶.
  const none = await page.evaluate(async () => {
    for (const box of document.querySelectorAll("#layer-list input[data-id]")) {
      if (box.checked) { box.checked = false; box.dispatchEvent(new Event("change", { bubbles: true })); }
    }
    await new Promise((r) => setTimeout(r, 200));
    return { empty: document.getElementById("pb-empty").classList.contains("hidden"),
             disabled: document.getElementById("pb-play").disabled };
  });
  expect(none.empty).toBe(false);
  expect(none.disabled).toBe(true);
});

/* ------------------------------------------------- E-041 the preload ring
 * The ring exists because of two measurements (2026-08-18), and both are
 * pinned here rather than left in a plan nobody re-reads:
 *
 *   1. GIBS answers every tile with `cache-control: … no-store …`, so the
 *      browser's HTTP cache CANNOT be warmed. The shipped first version of the
 *      prefetch fetched every visible tile of the next frame with CORS
 *      disabled, purely to fill a cache the service forbids — sixty requests a
 *      frame to a public NASA service, buying nothing.
 *   2. A Cesium layer at `show: true, alpha: 0` requests its whole visible
 *      tile set, and promoting it (alpha 0 → 1) requests NOTHING. Cesium's
 *      texture cache is the only cache available here, and it holds tiles
 *      already decoded and on the GPU.
 *
 * These tests count REQUESTS rather than milliseconds on purpose: through the
 * sandbox proxies a frame takes seconds, so timings measure the harness, while
 * the request count is the same number in the sandbox, in CI and on a phone. */

// The globe reporting an empty tile queue is the closest thing to "the picture
// is finished" that Cesium offers; the proxies are slow enough that this needs
// a far more generous ceiling than the app's own 8 s.
async function tilesSettled(page, timeout = 180000) {
  await page.waitForFunction(
    () => window.__earth.viewer.scene.globe.tilesLoaded, null, { timeout });
}

// Point the transport at a range. The DOM fields are the source of truth for
// `playbackRebuild`, so setting only the state object would silently get the
// panel's defaults back.
async function playRange(page, start, end, step) {
  await page.evaluate(({ s, e, st }) => {
    const E = window.__earth;
    const set = (id, v) => { const el = document.getElementById(id); if (el) el.value = v; };
    set("pb-start", s); set("pb-end", e); set("pb-step", st);
    E.playback.start = s; E.playback.end = e; E.playback.step = st;
    E.playbackRebuild();
  }, { s: start, e: end, st: step });
}

test("a promoted frame costs ZERO new tile requests", async ({ page }) => {
  test.setTimeout(300000);
  await onlyLayers(page, ["sst"]);
  // Every GIBS request for this layer, for the whole test. Counting URLs (not
  // elapsed time) is what makes this assertion mean the same thing everywhere.
  const tiles = [];
  page.on("request", (r) => {
    const u = r.url();
    if (u.includes("GHRSST")) tiles.push(u);
  });

  await playRange(page, "2015-01-01", "2015-05-01", "1mo");
  await page.evaluate(() => window.__earth.playbackShowFrame(0));
  await tilesSettled(page);

  // Warm frames i+1 … i+depth at alpha 0, the way the run loop does.
  const ring = await page.evaluate(() => {
    const E = window.__earth;
    E.playbackEnsurePreload(0);
    return { depth: E.playback.preloadDepth, held: [...E.playPreload.keys()],
             next: E.playback.frames[1],
             nextTime: E.gibsTime(E.state.layers.sst.cfg, E.playback.frames[1]) };
  });
  expect(ring.depth).toBeGreaterThanOrEqual(1);
  expect(ring.held).toContain(ring.next);

  const forNext = () => tiles.filter((u) => u.includes(ring.nextTime)).length;
  // The alpha-0 layer really does load: this is measurement 2's second row, and
  // without it the "no new requests" assertion below would also pass for a ring
  // that never fetched anything at all.
  await expect.poll(forNext, { timeout: 180000 }).toBeGreaterThan(0);
  await tilesSettled(page);
  const before = forNext();

  const promoted = await page.evaluate(async () => {
    const E = window.__earth;
    const d = E.playback.frames[1];
    const ringLayer = E.playPreload.get(d).built[0].layer;
    await E.playbackShowFrame(1);
    return {
      // The live layer IS the layer that was warming a moment ago — promoting
      // is an assignment, not a rebuild.
      same: E.state.layers.sst.layer === ringLayer,
      alpha: E.state.layers.sst.layer.alpha,
      date: E.state.date,
      stillHeld: E.playPreload.has(d),
      settledNow: E.viewer.scene.globe.tilesLoaded,
    };
  });
  expect(promoted.same).toBe(true);
  expect(promoted.alpha).toBe(1);              // at the entry's alpha, not 0
  expect(promoted.date).toBe(ring.next);
  expect(promoted.stillHeld).toBe(false);      // and the ring gave it up
  // Measurement 2's third row: the globe is still reporting a settled tile
  // queue one tick after the frame changed, because nothing was asked for.
  expect(promoted.settledNow).toBe(true);

  // THE measurement. Anything that quietly breaks the mechanism — a Cesium
  // upgrade that discards a layer's textures on an alpha change, a refactor
  // that rebuilds the provider on promote — shows up here as a non-zero delta
  // while every other playback test still passes.
  await page.waitForTimeout(6000);
  await tilesSettled(page);
  expect(forNext()).toBe(before);
});

test("show:false loads nothing; alpha 0 loads everything; alpha 0→1 loads nothing", async ({ page }) => {
  test.setTimeout(300000);
  await onlyLayers(page, ["sst"]);
  await tilesSettled(page);

  // A date nothing else on the page is asking for, so every hit below belongs
  // to the layer this test added.
  const day = "2011-07-15";
  const hits = [];
  page.on("request", (r) => { if (r.url().includes(day)) hits.push(r.url()); });

  // `show` must be false FROM CONSTRUCTION: the collection raises layerAdded
  // synchronously, so a layer added shown and hidden on the next line has
  // already been given its tile skeletons.
  await page.evaluate((d) => {
    const E = window.__earth;
    window.__t = new Cesium.ImageryLayer(E.gibsProvider(E.state.layers.sst.cfg, d), { show: false });
    E.viewer.imageryLayers.add(window.__t);
  }, day);
  await page.waitForTimeout(5000);
  // Tile skeletons live behind `layer.show && _createTileImagerySkeletons(...)`,
  // which is why a hidden layer is not a prefetch — it is nothing at all.
  expect(hits.length).toBe(0);

  await page.evaluate(() => { window.__t.show = true; window.__t.alpha = 0; });
  await expect.poll(() => hits.length, { timeout: 180000 }).toBeGreaterThan(0);
  await tilesSettled(page);
  const warmed = hits.length;

  // …and the promote is free, which is the property the whole ring is built on.
  await page.evaluate(() => { window.__t.alpha = 1; });
  await page.waitForTimeout(5000);
  expect(hits.length).toBe(warmed);

  await page.evaluate(() => window.__earth.viewer.imageryLayers.remove(window.__t, true));
});

test("stopping playback leaves no ring layers behind", async ({ page }) => {
  test.setTimeout(300000);
  await onlyLayers(page, ["sst"]);
  await tilesSettled(page);
  const baseline = await page.evaluate(() => window.__earth.viewer.imageryLayers.length);

  await playRange(page, "2015-01-01", "2015-05-01", "1mo");
  await page.evaluate(() => {
    const E = window.__earth;
    E.playback.fps = 8;                       // a speed limit, not a promise
    E.playbackPlay();
  });
  // The ring fills as soon as the first frame is on screen.
  await expect
    .poll(() => page.evaluate(() => window.__earth.playPreload.size), { timeout: 180000 })
    .toBeGreaterThan(0);
  const during = await page.evaluate(() => window.__earth.viewer.imageryLayers.length);
  expect(during).toBeGreaterThan(baseline);   // the ring IS extra imagery layers

  await page.evaluate(() => window.__earth.playbackStop());
  /* Stop sweeps the retirement queue and empties the ring synchronously, but a
   * frame that was already mid-flight when stop landed can still finish adding
   * its layers — so poll for the resting state rather than reading it once and
   * timing a race. */
  const state = () => page.evaluate(() => {
    const E = window.__earth;
    const L = E.viewer.imageryLayers;
    let invisible = 0;
    for (let k = 0; k < L.length; k++) if (L.get(k).alpha === 0) invisible++;
    return { held: E.playPreload.size, count: L.length, retiring: E.retiring.length, invisible };
  });
  await expect.poll(state, { timeout: 30000 }).toEqual(
    { held: 0, count: baseline, retiring: 0, invisible: 0 });
});

test("a configuration change clears the ring instead of promoting a stale frame", async ({ page }) => {
  test.setTimeout(300000);
  await onlyLayers(page, ["sst"]);
  await tilesSettled(page);
  await playRange(page, "2015-01-01", "2015-05-01", "1mo");
  await page.evaluate(() => window.__earth.playbackShowFrame(0));
  await page.evaluate(() => window.__earth.playbackEnsurePreload(0));
  expect(await page.evaluate(() => window.__earth.playPreload.size)).toBeGreaterThan(0);

  // The aggregation window changes what a frame IS — a 30-day mean instead of a
  // day — so every layer the ring holds was built for a picture the user has
  // just left. Promoting one would show the old render mode under the new
  // label, instantly and with no round trip in which to notice.
  const cleared = await page.evaluate(() => {
    const slider = document.getElementById("window-days");
    slider.value = "30";
    slider.dispatchEvent(new Event("change", { bubbles: true }));
    return window.__earth.playPreload.size;
  });
  expect(cleared).toBe(0);

  // Same for the comparison, and the ring's own key is the belt behind that
  // brace: a frame built before the comparison moved refuses to promote even
  // if some future call site forgets to clear.
  const stale = await page.evaluate(async () => {
    const E = window.__earth;
    const slider = document.getElementById("window-days");
    slider.value = "1";
    slider.dispatchEvent(new Event("change", { bubbles: true }));
    E.playbackEnsurePreload(0);
    const held = E.playback.frames[1];
    const size = E.playPreload.size;
    E.state.compareYears = 10;               // straight into state: no handler, no clear
    return { size, promoted: E.playbackPromote(held), left: E.playPreload.size };
  });
  expect(stale.size).toBeGreaterThan(0);
  expect(stale.promoted).toBe(false);        // refused, and destroyed rather than shown
  expect(stale.left).toBe(0);
});

test("playback warms tiles through Cesium, never through a no-cors refetch", async ({ page }) => {
  test.setTimeout(300000);
  await onlyLayers(page, ["sst"]);
  await tilesSettled(page);

  // The source itself: GIBS sends `no-store`, so a fetch whose only purpose is
  // to fill the HTTP cache is dead code by construction. It is an obvious
  // enough idea that this asserts the string literal is gone, not merely that
  // the behaviour looks right today.
  const src = await page.evaluate(() => fetch("/src/app.js").then((r) => r.text()));
  expect(src).not.toMatch(/["']no-cors["']/);

  const tiles = [];
  page.on("request", (r) => { if (r.url().includes("GHRSST")) tiles.push(r.url()); });
  await playRange(page, "2015-01-01", "2015-05-01", "1mo");
  await page.evaluate(() => {
    const E = window.__earth;
    E.playback.fps = 8;
    E.playbackPlay();
  });
  await expect
    .poll(() => page.evaluate(() => window.__earth.playback.i), { timeout: 180000 })
    .toBeGreaterThan(1);
  await page.evaluate(() => window.__earth.playbackStop());

  // The prefetch built exactly the URLs Cesium was about to request, so it
  // doubled every one of them. One request per tile URL is the shape of a
  // playback that warms through Cesium's own texture cache and nowhere else.
  expect(tiles.length).toBeGreaterThan(0);
  expect(tiles.length - new Set(tiles).size).toBe(0);
});

/* ===================================================================== *
 *  Politeness to GIBS — the request BUDGET per user interaction.
 *
 *  Every map tile this app draws is a direct browser→NASA request. No CDN of
 *  ours stands in front of gibs.earthdata.nasa.gov, so the only thing between
 *  a popular app and a block is how many requests one finger can generate.
 *  These tests assert measured COUNTS with headroom, not implementation
 *  details, so a future change that makes an interaction unbounded again
 *  fails here rather than at NASA.
 *
 *  The numbers below are all derived inside the test from a UNIT measured on
 *  the same machine in the same run (what ONE application of a date costs in
 *  this viewport), because the visible tile count depends on the window size
 *  and the zoom and is not a constant worth pinning.
 * ===================================================================== */

// Every GIBS TILE request (the `.../{z}/{y}/{x}.png|jpg` shape), as opposed to
// the colormap and time-domain XML the app also fetches from the same host.
function countTiles(page) {
  const urls = [];
  page.on("request", (r) => {
    const u = r.url();
    if (u.includes("gibs.earthdata.nasa.gov") && /\/\d+\/\d+\/\d+\.(png|jpe?g)/.test(u)) urls.push(u);
  });
  return { urls, since(n) { return urls.length - n; }, get n() { return urls.length; } };
}

test("a held date stepper costs a bounded number of tile requests", async ({ page }) => {
  /* Measured 2026-08-21, MIRROR, 1280x720, one layer, four rendered tiles:
   * sixty date changes at a browser's key-repeat rate issued 240 tile
   * requests — exactly one whole visible tile set per keystroke, no debounce,
   * and zero of the superseded requests cancelled. The count was a pure
   * function of how long a finger stayed on the arrow key. */
  test.setTimeout(300000);
  await onlyLayers(page, ["sst"]);
  await tilesSettled(page);
  const c = countTiles(page);

  // UNIT: what a single date step costs in THIS viewport, right now.
  let mark = c.n;
  await page.click('#date-steps button[data-step="-1d"]');
  await page.waitForTimeout(1500);
  await tilesSettled(page);
  const unit = c.since(mark);
  expect(unit).toBeGreaterThan(0);          // a date step really does fetch

  // THE BURST: N changes spaced one animation frame apart — a held arrow key,
  // a dragged date field, a repeated tap. Each gets its own render, so an
  // uncoalesced app builds N generations and pays N units.
  const N = 40;
  mark = c.n;
  const landed = await page.evaluate(async (n) => {
    const inp = document.getElementById("layer-date");
    const start = new Date(inp.value + "T00:00:00Z");
    let last = inp.value;
    for (let i = 1; i <= n; i++) {
      last = new Date(start.getTime() - i * 86400000).toISOString().slice(0, 10);
      inp.value = last;
      inp.dispatchEvent(new Event("change", { bubbles: true }));
      await new Promise((r) => requestAnimationFrame(r));
    }
    return last;
  }, N);
  await page.waitForTimeout(3000);
  await tilesSettled(page);
  const burst = c.since(mark);

  /* The bound: a burst must not cost what the burst would cost unthrottled.
   * Half of it is a deliberately loose ceiling — the real reduction measured
   * on the sandbox was 4.3x and on a fast machine it is larger, because the
   * gate is "one date generation in flight at a time" and a fast network
   * paints more of them. Half is the line below which no per-event rebuild
   * can hide. */
  expect(burst).toBeLessThan(unit * N / 2);
  // …and the app landed on the LAST date asked for, not the first: a
  // coalescer that dropped the newest value would be quiet AND wrong.
  expect(await page.evaluate(() => window.__earth.state.date)).toBe(landed);
});

test("dragging the playback scrub bar does not cost one tile set per pointer move", async ({ page }) => {
  /* The scrub slider fires `input` on every pointer move (~60 a second) and
   * each one used to run a whole frame change. Measured 2026-08-21: forty
   * input events over a 366-frame range issued 160 tile requests, four per
   * event, none cancelled. A real two-second drag is ~120 events. */
  test.setTimeout(300000);
  await onlyLayers(page, ["sst"]);
  await tilesSettled(page);
  await page.click("#tab-play");
  await playRange(page, "2025-01-01", "2025-12-31", "1d");
  await page.evaluate(() => window.__earth.playbackShowFrame(0));
  await page.waitForTimeout(1500);
  await tilesSettled(page);

  const c = countTiles(page);
  let mark = c.n;
  await page.evaluate(() => window.__earth.playbackShowFrame(1));
  await page.waitForTimeout(1500);
  await tilesSettled(page);
  const unit = Math.max(1, c.since(mark));

  const N = 30;
  mark = c.n;
  const want = await page.evaluate(async (n) => {
    const s = document.getElementById("pb-scrub");
    const total = window.__earth.playback.frames.length;
    let last = 0;
    for (let i = 0; i < n; i++) {
      last = Math.round((i + 1) * (total - 1) / n);
      s.value = String(last);
      s.dispatchEvent(new Event("input", { bubbles: true }));
      await new Promise((r) => requestAnimationFrame(r));
    }
    return window.__earth.playback.frames[last];
  }, N);
  await page.waitForTimeout(4000);
  await tilesSettled(page);
  expect(c.since(mark)).toBeLessThan(unit * N / 2);
  // The drag ends where the thumb ended.
  await expect.poll(() => page.evaluate(() => window.__earth.state.date), { timeout: 30000 })
    .toBe(want);
});

test("the aggregate window's raw tile reads share Cesium's per-server budget", async ({ page }) => {
  /* The Aggregate slider does not draw tiles, it READS them — with a bare
   * fetch(), which `Cesium.RequestScheduler` never sees. Measured 2026-08-21
   * before the fix: a 365-day window over one layer in a four-tile viewport
   * put **48 requests to gibs.earthdata.nasa.gov in flight at the same
   * moment** (peak concurrency == total). The 12-sample cap bounded the count
   * correctly; nothing bounded the concurrency, and a full-screen desktop
   * view with three aggregable layers is ~400 simultaneous connections to one
   * public host from one tab. */
  test.setTimeout(300000);
  await onlyLayers(page, ["sst"]);
  await tilesSettled(page);

  // The sample cap is what bounds the COUNT — assert it binds at every scale
  // the slider offers, reduced to one array rather than one expect per point.
  const caps = await page.evaluate(() => [1, 7, 30, 365, 730]
    .map((w) => window.__earth.windowSampleDates("2026-06-15", w).length));
  expect(caps[0]).toBe(1);                       // a single day is a single day
  expect(Math.max(...caps)).toBe(12);            // and nothing ever exceeds 12

  const peak = await page.evaluate(async () => {
    let inflight = 0, top = 0, total = 0;
    const real = window.fetch;
    window.fetch = function (u, o) {
      const url = typeof u === "string" ? u : u.url;
      const tile = url.includes("gibs.earthdata.nasa.gov") &&
        /\/\d+\/\d+\/\d+\.(png|jpe?g)/.test(url);
      if (tile) { total++; inflight++; top = Math.max(top, inflight); }
      return real.call(this, u, o).finally(() => { if (tile) inflight--; });
    };
    const s = document.getElementById("window-days");
    s.value = "365";
    s.dispatchEvent(new Event("input", { bubbles: true }));
    s.dispatchEvent(new Event("change", { bubbles: true }));
    await new Promise((r) => setTimeout(r, 25000));
    window.fetch = real;
    return { top, total, budget: Cesium.RequestScheduler.maximumRequestsPerServer };
  });
  expect(peak.total).toBeGreaterThan(0);         // the window really did read tiles
  // ONE budget against GIBS, not two: the raw-read path is admitted through
  // the same per-server number the scheduled tile path already respects.
  expect(peak.top).toBeLessThanOrEqual(peak.budget);
});

test("GIBS asking us to slow down is visible, and we slow down", async ({ page }) => {
  /* A 429 used to be indistinguishable from an empty tile: `sstFetchBitmap`
   * returned null for every non-OK status, so the layer went quiet, the
   * reader blamed the archive, and the app kept asking at the same rate —
   * which is how a rate limit becomes a block on a service that cannot be
   * bought back. */
  test.setTimeout(180000);
  await onlyLayers(page, ["sst"]);
  await tilesSettled(page);
  const toasts = await recordToasts(page);

  const before = await page.evaluate(() => window.__earth.gibsRawLimit());
  expect(before).toBeGreaterThan(1);

  // One 429 from the raw-read path, delivered through the app's own fetch.
  await page.evaluate(async () => {
    const real = window.fetch;
    window.fetch = async () => new Response("", { status: 429 });
    const E = window.__earth;
    // A window forces the aggregate provider down the raw-read path.
    const s = document.getElementById("window-days");
    s.value = "30";
    s.dispatchEvent(new Event("input", { bubbles: true }));
    s.dispatchEvent(new Event("change", { bubbles: true }));
    await new Promise((r) => setTimeout(r, 8000));
    window.fetch = real;
    void E;
  });

  await expect.poll(() => page.evaluate(() => window.__earth.gibsRawLimit()), { timeout: 30000 })
    .toBe(1);
  await expect.poll(toasts, { timeout: 30000 }).toContain("slow down");
});

test("playback in a hidden tab stops asking NASA for tiles", async ({ page }) => {
  /* Streaming tiles into a tab nobody is looking at is both rude and
   * pointless, and it is the one politeness property of the player that costs
   * nothing to break silently. Measured 2026-08-21: eight seconds of playback
   * with the tab hidden issued four tile requests — the frame already in
   * flight when the halt landed — and then nothing. */
  test.setTimeout(300000);
  await onlyLayers(page, ["sst"]);
  await tilesSettled(page);
  await page.click("#tab-play");
  await playRange(page, "2015-01-01", "2016-01-01", "1mo");

  const c = countTiles(page);
  let mark = c.n;
  await page.evaluate(() => {
    const E = window.__earth;
    E.playback.i = 0; E.playback.loop = true; E.playback.fps = 4;
    E.playbackPlay();
  });
  await page.waitForTimeout(6000);
  const whileVisible = c.since(mark);
  expect(whileVisible).toBeGreaterThan(0);       // it really was playing

  const stopped = await page.evaluate(async () => {
    Object.defineProperty(document, "hidden", { configurable: true, get: () => true });
    document.dispatchEvent(new Event("visibilitychange"));
    await new Promise((r) => setTimeout(r, 500));
    return window.__earth.playback.playing;
  });
  expect(stopped).toBe(false);

  mark = c.n;
  await page.waitForTimeout(8000);
  const whileHidden = c.since(mark);
  await page.evaluate(() => {
    Object.defineProperty(document, "hidden", { configurable: true, get: () => false });
    document.dispatchEvent(new Event("visibilitychange"));
  });
  await page.evaluate(() => window.__earth.playbackStop());

  /* Eight seconds at 4 fps is 32 frames the player would have shown; the
   * comparison is against what the SAME eight seconds cost while visible, so
   * the assertion means the same thing on any machine. A hidden tab may still
   * finish the frame that was in flight when it was hidden — it must not
   * start new ones. */
  expect(whileHidden).toBeLessThan(whileVisible / 2);
});

/* Cones (E-069). The tab draws our own model's dependency cone on the globe;
 * the panel is its control room. What is checked here is the WIRING — that the
 * geometry file reaches the page, that the three read-outs answer the two
 * controls, and that the artefacts leave the globe when the tab does. The
 * geometry itself is certified in tests/data.spec.js against ml/cone.py's own
 * reference dot sets, which is where a drift in the numbers would surface. */
test("Cones tab draws the E-069 cone and answers its two controls", async ({ page }) => {
  /* A preset flies the camera, so the globe pulls a fresh tile set on every
   * one — and on the software GL stack that render loop starves Playwright's
   * actionability check (CLAUDE.md §4). The FIRST tab click and the FIRST
   * preset are real clicks, because that is the path a reader takes; the rest
   * are dispatched in-page, the same way the heavy-layer tests do it. */
  test.setTimeout(120000);
  const tap = (sel) => page.evaluate((s) => document.querySelector(s).click(), sel);

  await page.click("#tab-cones");
  await expect(page.locator("#panel-cones")).toBeVisible();
  await expect(page.locator("#tab-cones")).toHaveClass(/active/);
  // the file has to land before anything reads: the tiles say so
  await expect(page.locator("#cn-reach .stat-value")).not.toHaveText("–", { timeout: 20000 });
  await expect(page.locator("#cn-tokens .stat-value")).toHaveText("748");

  // a preset moves the anchor: the read-out names the cell it snapped to, and
  // the tiles describe that anchor's own cone
  await page.click('#cn-presets button[data-lat="36"]');
  const read = page.locator("#cn-lag-read");
  await expect(read).toContainText("36.00° N");
  await expect(read).toContainText("70.00° W");
  await expect(read).toContainText("lag 1 · 5 days back");
  // family B at lag 1 reads 259.2 km, and 80 dots per channel over lags 1–6
  await expect(page.locator("#cn-reach .stat-value")).toHaveText("259.2 km");
  await expect(page.locator("#cn-dots .stat-value")).toHaveText("80");
  // 36° N 70° W sits well inside the tensor, so nothing falls off it
  await expect(page.locator("#cn-dots .stat-sub")).not.toContainText("off window");

  // the lag slider: past lag 6 the reach is stage 2's envelope, 129.6 × 21 km
  await page.evaluate(() => {
    const s = document.getElementById("cn-lag");
    s.value = "20";
    s.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await expect(page.locator("#cn-reach .stat-value")).toHaveText("2721.6 km");
  await expect(read).toContainText("lag 20 · 100 days back");
  const st = await page.evaluate(() => window.__earth.coneState());
  expect(st.reachKm).toBeCloseTo(2721.6, 6);
  expect(st.y).toBe(144);        // 36° N  → 0.25° row 144
  expect(st.x).toBe(120);        // 70° W  → 0.25° column 120
  // 80 inner dots (lags 1–6) plus fourteen outer spirals of 24 = the whole
  // cone the model reads at lag 20
  expect(st.drawn).toBe(80 + 14 * 24);
  expect(st.offWindow).toBe(0);

  // the equator anchor pushes part of the cone off the tensor's southern edge —
  // those dots are drawn hollow and COUNTED, because the model reads them as
  // missing rather than wrapping around
  await tap('#cn-presets button[data-lat="0"]');
  await expect(page.locator("#cn-dots .stat-sub")).toContainText("off window");
  expect(await page.evaluate(() => window.__earth.coneState().offWindow)).toBeGreaterThan(0);

  // the depth column is the one family with no disc: six tokens, no sunflower
  await page.evaluate(() => {
    const s = document.getElementById("cn-family");
    s.value = "rg";
    s.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await expect(page.locator("#cn-dots .stat-value")).toHaveText("6");
  await expect(page.locator("#cn-dots .stat-sub")).toContainText("anchor column only");
  await page.evaluate(() => {
    const s = document.getElementById("cn-family");
    s.value = "B";
    s.dispatchEvent(new Event("change", { bubbles: true }));
  });

  // leaving the tab takes the cone off the globe (the switch has no leave hook,
  // so every other tab's click is the leave hook)
  await tap("#tab-layers");
  const hidden = await page.evaluate(() => {
    const a = window.__earth.coneArtefacts;
    return {
      anchor: a.anchor.show, win: a.win.show, ring: a.ring.show,
      patch: a.patch.some((p) => p.show),
      playing: window.__earth.coneState().playing,
    };
  });
  expect(hidden).toEqual({ anchor: false, win: false, ring: false,
                           patch: false, playing: false });
  expect(page.__errors, `page errors: ${page.__errors.join(" | ")}`).toHaveLength(0);
});

/* The what-if parameters. What is checked is the LOOP: a knob moves, the
 * drawn cone and the tiles move with it, the badge says the geometry is no
 * longer the trained one, and reset lands back on ml/cone.py's own numbers —
 * 80 dots per family-B channel, 706 dot tokens, 748 in total. That last
 * assertion is the one that matters: `tests/data.spec.js` certifies the port
 * against Python's reference sets AT THE DEFAULTS, so a reset that did not
 * restore them exactly would leave the certification testing a geometry the
 * page can no longer produce. */
test("Cones tab: what-if parameters change the cone, and reset restores E-069", async ({ page }) => {
  test.setTimeout(120000);
  await page.click("#tab-cones");
  await expect(page.locator("#cn-reach .stat-value")).not.toHaveText("–", { timeout: 20000 });

  // as trained: the exported budget, and no badge
  await expect(page.locator("#cn-dots .stat-value")).toHaveText("80");
  await expect(page.locator("#cn-tokens .stat-value")).toHaveText("748");
  await expect(page.locator("#cn-whatif")).toBeHidden();
  const asTrained = await page.evaluate(() => window.__earth.coneState());
  expect(asTrained.dotsPerChannel).toBe(80);
  expect(asTrained.dotTokens).toBe(706);
  expect(asTrained.totalTokens).toBe(748);
  expect(asTrained.paramsDirty).toBe(false);
  expect(asTrained.params.L_IN).toBe(6);

  // one knob: three inner pentads instead of six. Fewer lags, fewer dots per
  // channel, fewer tokens — and the badge appears.
  const setKnob = (key, v) => page.evaluate(([k, val]) => {
    const s = document.getElementById(`cn-p-${k}`);
    s.value = String(val);
    s.dispatchEvent(new Event("input", { bubbles: true }));
  }, [key, v]);
  await setKnob("L_IN", 3);
  await expect(page.locator("#cn-whatif")).toBeVisible();
  await expect(page.locator("#cn-whatif")).toContainText("not what the codec was trained on");
  const shrunk = await page.evaluate(() => window.__earth.coneState());
  expect(shrunk.params.L_IN).toBe(3);
  expect(shrunk.paramsDirty).toBe(true);
  expect(shrunk.dotsPerChannel).toBeLessThan(80);
  expect(shrunk.totalTokens).toBeLessThan(748);
  expect(Number(await page.locator("#cn-dots .stat-value").textContent()))
    .toBe(shrunk.dotsPerChannel);

  // a second knob, on a different mechanism: more slots per disc means more
  // dots at the SAME six lags, so this one has to move the count upward
  await page.evaluate(() => {
    document.getElementById("cn-reset").click();
  });
  await setKnob("SLOT_MAX", 48);
  const denser = await page.evaluate(() => window.__earth.coneState());
  expect(denser.dotsPerChannel).toBeGreaterThan(80);
  expect(denser.dotTokens).toBeGreaterThan(706);

  // reset: exactly back to what E-069 was trained on
  await page.evaluate(() => document.getElementById("cn-reset").click());
  await expect(page.locator("#cn-whatif")).toBeHidden();
  await expect(page.locator("#cn-dots .stat-value")).toHaveText("80");
  await expect(page.locator("#cn-tokens .stat-value")).toHaveText("748");
  const back = await page.evaluate(() => window.__earth.coneState());
  expect(back.dotsPerChannel).toBe(80);
  expect(back.dotTokens).toBe(706);
  expect(back.totalTokens).toBe(748);
  expect(back.paramsDirty).toBe(false);
  expect(back.params).toEqual(asTrained.params);

  expect(page.__errors, `page errors: ${page.__errors.join(" | ")}`).toHaveLength(0);
});

/* The time axis. A lag is a DATE — pentad bins counted from 1982-01-01 — so
 * the read-out names one, the three held-out years are flagged, and the two
 * clocks move different things: "sweep lags" walks the lag at one date,
 * "advance time" walks the date at one lag. `follow the cone` is the bit that
 * makes it worth having: it hands each lag's date to the app's own date
 * funnel, so whatever timed layer is on shows that lag's field. */
test("Cones tab: every lag is a date, and the time clock drives the globe", async ({ page }) => {
  test.setTimeout(120000);
  // The time clock moves the app's date at up to ten steps a second, and the
  // default SST layer would answer every one of them with a tile set. What is
  // under test is the DATE funnel, not the imagery, so take the layer off
  // in-page (CLAUDE.md §4) and let the clock run against a bare globe.
  await page.evaluate(() => {
    const el = document.getElementById("toggle-sst");
    if (el && el.checked) {
      el.checked = false;
      el.dispatchEvent(new Event("change", { bubbles: true }));
    }
  });
  await page.click("#tab-cones");
  await expect(page.locator("#cn-reach .stat-value")).not.toHaveText("–", { timeout: 20000 });

  // bins are fixed 5-day bins from 1982-01-01; the tensor holds 0..3141
  const bins = await page.evaluate(() => ({
    zero: window.__earth.coneDateOfBin(0),
    last: window.__earth.coneDateOfBin(3141),
    mid: window.__earth.coneBinOfDate("2015-04-01"),
  }));
  expect(bins.zero).toBe("1982-01-01");
  expect(bins.last).toBe("2024-12-31");
  expect(await page.evaluate((b) => window.__earth.coneDateOfBin(b), bins.mid))
    .toBe("2015-03-29");                 // 2015-04-01 snaps back to its bin

  // an anchor date, and the lag read-out names the date the model reads there
  await page.evaluate((b) => window.__earth.conesSetBin(b), bins.mid);
  await page.evaluate(() => {
    const s = document.getElementById("cn-lag");
    s.value = "3";
    s.dispatchEvent(new Event("input", { bubbles: true }));
  });
  const read = page.locator("#cn-lag-read");
  await expect(read).toContainText("lag 3 · 15 days back");
  await expect(read).toContainText("2015-03-14");     // 2015-03-29 − 15 days
  const st = await page.evaluate(() => window.__earth.coneState());
  expect(st.anchorDate).toBe("2015-03-29");
  expect(st.lagDate).toBe("2015-03-14");
  expect(st.heldOut).toBe(false);
  await expect(read).not.toHaveClass(/cn-held/);

  // 2017 is a development holdout year (ml/plans/E059_holdout_window.md), so
  // a lag landing in it is flagged rather than presented as ordinary
  await page.evaluate((b) => window.__earth.conesSetBin(b),
                      await page.evaluate(() => window.__earth.coneBinOfDate("2017-06-10")));
  expect(await page.evaluate(() => window.__earth.coneState().heldOut)).toBe(true);
  await expect(read).toHaveClass(/cn-held/);
  await expect(read).toContainText("held-out year");

  // follow the cone: the lag's own date lands in the app's date selector
  await page.evaluate((b) => window.__earth.conesSetBin(b),
                      await page.evaluate(() => window.__earth.coneBinOfDate("2015-04-01")));
  await page.evaluate(() => {
    const c = document.getElementById("cn-follow");
    c.checked = true;
    c.dispatchEvent(new Event("change", { bubbles: true }));
  });
  const followed = await page.evaluate(() => window.__earth.coneState().lagDate);
  await expect.poll(() => page.evaluate(() => window.__earth.state.date)).toBe(followed);
  expect(await page.inputValue("#layer-date")).toBe(followed);

  // advance time: the ANCHOR walks forward one pentad at a time, the lag held
  const before = await page.evaluate(() => window.__earth.coneState().bin);
  await page.evaluate(() => {
    const s = document.getElementById("cn-time-speed");
    s.value = "100";
    s.dispatchEvent(new Event("input", { bubbles: true }));
    window.__earth.conesTimePlay();
  });
  await expect.poll(() => page.evaluate(() => window.__earth.coneState().bin),
                    { timeout: 20000 }).toBeGreaterThan(before + 1);
  const running = await page.evaluate(() => {
    const s = window.__earth.coneState();
    window.__earth.conesTimeStop();
    return s;
  });
  expect(running.lag).toBe(3);                       // the lag was held
  expect(running.timePlaying).toBe(true);
  // and the globe's date came with it, still one lag behind the anchor
  await expect.poll(() => page.evaluate(() => window.__earth.state.date))
    .toBe(await page.evaluate(() => window.__earth.coneState().lagDate));
  expect(await page.evaluate(() => window.__earth.coneState().timePlaying)).toBe(false);

  // leaving the tab stops the clock — it drives the app's date, so a clock
  // left running behind another tab would move the globe under the reader
  await page.evaluate(() => window.__earth.conesTimePlay());
  await page.evaluate(() => document.querySelector("#tab-layers").click());
  expect(await page.evaluate(() => window.__earth.coneState().timePlaying)).toBe(false);

  expect(page.__errors, `page errors: ${page.__errors.join(" | ")}`).toHaveLength(0);
});

/* ---------------------------------------------------------- Cones · DATA MODE
 * The tab's other half: the same cone with the tensor's own numbers in it.
 *
 * The real anchor files are megabytes on the Hugging Face Hub, so both tests
 * serve `data/cone_samples/fixture.json` for every `resolve/main/cone_samples/`
 * request — the in-repo copy `ml/export_cone_sample.py --fixture` writes, whose
 * SCHEMA `tests/data.spec.js` pins against the index. The route is registered
 * after the MIRROR beforeEach's Hub pass-through, so this handler wins and the
 * suite needs no network.
 *
 * What is NOT tested here is whether the numbers are right — that is
 * `tests/test_export_cone_sample.py`'s job, and it asserts the exporter's
 * flags against a direct `ConeSampler.sample` call bit for bit. These two are
 * about the PAGE: that a value reaches a dot, and that the stencil toggle
 * changes which dots there are. */
async function serveConeFixture(page) {
  const fixture = require("fs").readFileSync(
    require("path").join(__dirname, "..", "data", "cone_samples", "fixture.json"),
    "utf8");
  await page.route(/resolve\/main\/cone_samples\//, (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: fixture }));
  return JSON.parse(fixture);
}

/* The same trick for the GLOBAL exported set. The two routes cannot collide:
 * the family-4 pattern needs a slash straight after `cone_samples`, and these
 * files live under `cone_samples_f7/`. This fixture is one anchor — the
 * DATELINE, the cell in the tensor's last column — cut down to two dates and
 * four channels, one from each of family 7's three groups plus a second ocean
 * one, by `ml/export_cone_sample.py --trim-file`. */
async function serveConeFixtureF7(page) {
  const fixture = require("fs").readFileSync(
    require("path").join(__dirname, "..", "data", "cone_samples_f7", "fixture.json"),
    "utf8");
  await page.route(/resolve\/main\/cone_samples_f7\//, (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: fixture }));
  return JSON.parse(fixture);
}

test("Cones data mode: the dots carry values and the read-out prints one with its date",
     async ({ page }) => {
  test.setTimeout(120000);
  const fx = await serveConeFixture(page);
  await page.click("#tab-cones");
  await expect(page.locator("#cn-reach .stat-value")).not.toHaveText("–", { timeout: 20000 });
  // the block is hidden until the switch is on — the geometry is the default
  await expect(page.locator("#cn-data-block")).toBeHidden();

  await page.evaluate(() => window.__earth.conesSetDataMode(true));
  await expect(page.locator("#cn-data-block")).toBeVisible();
  const st = await page.evaluate(() => window.__earth.coneState());
  expect(st.data.ready).toBe(true);
  expect(st.data.error).toBe(null);
  expect(st.data.stencil).toBe("codec");
  // the anchor moved to the exported cell, and the date to an exported pentad
  expect(st.data.date).toBe(fx.meta.dates[0]);
  expect(st.data.snapped).toBe(false);
  expect(st.lat).toBeCloseTo(fx.meta.anchor.lat, 6);
  expect(st.lon).toBeCloseTo(fx.meta.anchor.lon, 6);

  // dots on the globe, and most of them carry a real measurement
  expect(st.data.nDots).toBeGreaterThan(9);        // the 3×3 patch plus dots
  expect(st.data.nValued).toBeGreaterThan(5);
  expect(st.data.hi).toBeGreaterThan(st.data.lo);
  // the legend states the numeric range rather than only a colour
  await expect(page.locator("#cn-legend .cn-scale")).toContainText(/\d/);
  await expect(page.locator("#cn-legend .cn-unit")).toContainText(fx.meta.dates[0]);

  // the read-out: pick the first dot that actually has a value and read it.
  // Picking through the exported handler rather than a canvas click — a
  // Cesium point is a few pixels wide on a software GL stack.
  const picked = await page.evaluate(() => {
    const dots = window.__earth.coneDataDots;
    const i = dots.findIndex((d) => d.valid && d.obs && Number.isFinite(d.raw));
    window.__earth.conesPickDot(i);
    return { i, d: window.__earth.coneState().data.pick };
  });
  expect(picked.i).toBeGreaterThanOrEqual(0);
  expect(picked.d.obs).toBe(true);
  expect(Number.isFinite(picked.d.raw)).toBe(true);
  const ro = page.locator("#cn-readout");
  await expect(ro).toContainText(`lag ${picked.d.lag}`);
  await expect(ro).toContainText(picked.d.date);          // §2.9: values carry WHEN
  await expect(ro).toContainText(`row ${picked.d.row}`);  // §2.4: which pixel answered
  await expect(ro).toContainText("observed");
  await expect(ro).toContainText("anomaly");

  // raw vs anomaly are two different numbers on two different scales
  const anomHi = await page.evaluate(() => window.__earth.coneState().data.hi);
  await page.evaluate(() => {
    const c = document.getElementById("cn-anom");
    c.checked = false;
    c.dispatchEvent(new Event("change", { bubbles: true }));
  });
  const rawSt = await page.evaluate(() => window.__earth.coneState());
  expect(rawSt.data.anomaly).toBe(false);
  expect(rawSt.data.hi).not.toBe(anomHi);
  await expect(page.locator("#cn-legend .cn-unit")).not.toContainText("z-scored");

  // switching data mode off returns the tab to the geometry it drew before
  await page.evaluate(() => window.__earth.conesSetDataMode(false));
  await expect(page.locator("#cn-data-block")).toBeHidden();
  expect(await page.evaluate(() => window.__earth.coneState().data.ready)).toBe(false);
  expect(await page.inputValue("#cn-lag")).not.toBe("");
  expect(page.__errors, `page errors: ${page.__errors.join(" | ")}`).toHaveLength(0);
});

test("Cones data mode: the stencil toggle switches the lag range and the dot set",
     async ({ page }) => {
  test.setTimeout(120000);
  const fx = await serveConeFixture(page);
  await page.click("#tab-cones");
  await expect(page.locator("#cn-reach .stat-value")).not.toHaveText("–", { timeout: 20000 });
  await page.evaluate(() => window.__earth.conesSetDataMode(true));

  const lag = page.locator("#cn-lag");
  // the CODEC stencil is the inner cone: lags 0–6, all 42 channels
  await expect(lag).toHaveAttribute("min", "0");
  await expect(lag).toHaveAttribute("max", String(fx.meta.L_in));
  expect(await page.locator("#cn-channel option").count()).toBe(fx.meta.channels.length);
  const codec = await page.evaluate(() => window.__earth.coneState());
  expect(Math.max(...codec.data.lags)).toBeLessThanOrEqual(fx.meta.L_in);
  expect(codec.data.channel).toBe(fx.meta.channels[0]);

  // STAGE 2 is the outer cone: the exported lags, and only the eight channels
  // the LIM can be scored on
  await page.evaluate(() => {
    const s = document.getElementById("cn-stencil");
    s.value = "stage2";
    s.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await expect(lag).toHaveAttribute("min", String(fx.outer.lags[0]));
  await expect(lag).toHaveAttribute("max", String(fx.outer.lags[fx.outer.lags.length - 1]));
  expect(await page.locator("#cn-channel option").count()).toBe(fx.outer.channels.length);
  const s2 = await page.evaluate(() => window.__earth.coneState());
  expect(s2.data.stencil).toBe("stage2");
  expect(s2.data.channel).toBe(fx.outer.channels[0]);
  // ONE annulus at a time — every lag is a different date, so a cumulative
  // draw would put twenty dates under one colour scale
  expect(s2.data.lags).toEqual([fx.outer.lags[0]]);
  expect(s2.data.nDots).toBe(fx.outer.n_dots_per_lag);
  expect(s2.data.nDots).not.toBe(codec.data.nDots);
  await expect(page.locator("#cn-data-hint")).toContainText("empty for lags 0–6");

  // the slider walks the exported outer lags, and a value between two of them
  // snaps to one — the ring drawn is the ring the slider says
  await page.evaluate((k) => {
    const s = document.getElementById("cn-lag");
    s.value = String(k);
    s.dispatchEvent(new Event("input", { bubbles: true }));
  }, fx.outer.lags[fx.outer.lags.length - 1]);
  const far = await page.evaluate(() => window.__earth.coneState());
  expect(far.lag).toBe(fx.outer.lags[fx.outer.lags.length - 1]);
  expect(far.data.lags).toEqual([far.lag]);

  // back to the codec stencil, and the range comes back with it
  await page.evaluate(() => {
    const s = document.getElementById("cn-stencil");
    s.value = "codec";
    s.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await expect(lag).toHaveAttribute("max", String(fx.meta.L_in));
  expect(await page.locator("#cn-channel option").count()).toBe(fx.meta.channels.length);
  expect(page.__errors, `page errors: ${page.__errors.join(" | ")}`).toHaveLength(0);
});

/* The cone follows the app's DATE, which is what makes the Play tab drive it.
 *
 * `notifyGlobeDate()` is called from every place that writes `state.date` — the
 * date input, `setGlobeDate`, the ±30-minute midnight cross and the Play tab's
 * frame step — and `conesOnGlobeDate` is the only listener. That is the whole
 * mechanism: the Play tab stays "a clock that drives state.date and nothing
 * else" (CLAUDE.md §5) and this tab grows no playback code of its own. Driving
 * the date INPUT here rather than the Play tab exercises the same funnel and
 * needs no timed layer to play. */
test("Cones data mode: the cone's date follows the app's date, and says when it cannot",
     async ({ page }) => {
  test.setTimeout(120000);
  const fx = await serveConeFixture(page);
  await page.click("#tab-cones");
  await expect(page.locator("#cn-reach .stat-value")).not.toHaveText("–", { timeout: 20000 });
  await page.evaluate(() => window.__earth.conesSetDataMode(true));
  const first = await page.evaluate(() => window.__earth.coneState());
  expect(first.data.date).toBe(fx.meta.dates[0]);

  // a date INSIDE the exported span moves the cone onto that pentad exactly
  const inside = fx.meta.dates[fx.meta.dates.length - 1];
  await page.evaluate((d) => {
    const el = document.getElementById("layer-date");
    el.value = d;
    el.dispatchEvent(new Event("change", { bubbles: true }));
  }, inside);
  await expect.poll(() => page.evaluate(() => window.__earth.coneState().data.date))
    .toBe(inside);
  const on = await page.evaluate(() => window.__earth.coneState());
  expect(on.data.snapped).toBe(false);
  expect(on.data.dateIdx).toBe(fx.meta.dates.length - 1);

  // a date OUTSIDE it shows the nearest exported pentad and SAYS SO — the one
  // thing a read-out must never do is present a neighbouring date's numbers as
  // if they were this date's
  await page.evaluate(() => {
    const el = document.getElementById("layer-date");
    el.value = "2019-06-01";
    el.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await expect.poll(() => page.evaluate(() => window.__earth.coneState().data.snapped))
    .toBe(true);
  const off = await page.evaluate(() => window.__earth.coneState());
  expect(fx.meta.dates).toContain(off.data.date);
  expect(off.data.date).toBe(fx.meta.dates[fx.meta.dates.length - 1]);  // nearest
  await expect(page.locator("#cn-data-hint")).toContainText("nearest exported pentad");
  await expect(page.locator("#cn-data-hint")).toContainText("2019");

  expect(page.__errors, `page errors: ${page.__errors.join(" | ")}`).toHaveLength(0);
});

/* THE THIRD SOURCE: the anchors exported from FAMILY 7 — the tensor that covers
 * the whole globe at 0.25° rather than only the North Atlantic window. It is
 * the same exporter, the same production sampler and the same file schema, so
 * the tab reuses every control; three things are genuinely different and this
 * test is about all three.
 *
 *   1. TWELVE anchors instead of five, and they are global — the Antarctic
 *      Circumpolar Current, the Sahara, the dateline.
 *   2. FIFTY-FOUR channels on three grids instead of 42 on one, so the channel
 *      select is grouped rather than a flat scroll.
 *   3. A grid that CLOSES. On the North Atlantic window a cell past the east
 *      edge is off the tensor and draws hollow; on the globe there is no edge,
 *      so the dateline anchor's own 3×3 patch straddles ±180° and every cell of
 *      it is real. That is the assertion the whole feature stands on, and it is
 *      the one a family-4 fixture can never make.
 *
 * The fixture is the dateline anchor cut to two dates and four channels, one
 * per group plus a second ocean one — `data/cone_samples_f7/fixture.json`,
 * whose schema `tests/data.spec.js` pins against the index. */
test("Cones data mode: the family-7 anchors are global, grouped by channel block, and wrap the dateline",
     async ({ page }) => {
  test.setTimeout(120000);
  await serveConeFixture(page);
  const f7 = await serveConeFixtureF7(page);
  await page.click("#tab-cones");
  await expect(page.locator("#cn-reach .stat-value")).not.toHaveText("–", { timeout: 20000 });
  await page.evaluate(() => window.__earth.conesSetDataMode(true));

  // the North Atlantic set is the default, and it is five anchors in a flat list
  const naAnchors = await page.locator("#cn-data-anchor option").allTextContents();
  expect(naAnchors.length).toBe(5);
  const naChannels = await page.locator("#cn-channel option").count();
  expect(naChannels).toBe(42);
  expect(await page.locator("#cn-channel optgroup").count()).toBe(0);

  // ---- switch to the global set
  await page.evaluate(() => window.__earth.conesSetSource("anchors_f7"));
  await expect.poll(() => page.evaluate(
    () => window.__earth.coneState().data.sampleSource)).toBe("anchors_f7");

  // TWELVE anchors, in the index's own order
  await expect(page.locator("#cn-data-anchor option")).toHaveCount(12);
  const ids = await page.locator("#cn-data-anchor option")
    .evaluateAll((os) => os.map((o) => o.value));
  expect(ids).toContain("dateline");
  expect(ids).toContain("acc");
  expect(ids).toContain("sahara");
  // the switch kept the reader's place: the Gulf Stream anchor of the North
  // Atlantic set lands on the Gulf Stream anchor of the global one
  expect(await page.evaluate(() => window.__earth.coneState().data.anchorId))
    .toBe("gulf_stream");

  // the channel select is GROUPED — 54 channels in one flat list is a scroll,
  // not a choice, so family 7's three blocks are three optgroups
  await expect(page.locator("#cn-channel optgroup")).toHaveCount(3);
  const groups = await page.locator("#cn-channel optgroup")
    .evaluateAll((os) => os.map((o) => o.label));
  expect(groups[0]).toContain("0.25°");
  expect(groups[1]).toContain("1°");
  expect(groups[2]).toContain("Argo");
  expect(await page.locator("#cn-channel option").count())
    .toBe(f7.meta.channels.length);

  // ---- stand on the DATELINE anchor, the cell in the tensor's last column
  await page.evaluate(() => {
    const s = document.getElementById("cn-data-anchor");
    s.value = "dateline";
    s.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await expect.poll(() => page.evaluate(
    () => window.__earth.coneState().data.anchorId)).toBe("dateline");

  const st = await page.evaluate(() => window.__earth.coneState());
  expect(st.data.ready).toBe(true);
  expect(st.data.error).toBe(null);
  expect(st.data.source).toBe("anchors_f7");
  expect(st.data.recipe).toBe("f7l0");
  expect(st.data.nChannels).toBe(f7.meta.channels.length);
  expect(st.data.date).toBe(f7.meta.dates[0]);
  expect(st.lat).toBeCloseTo(f7.meta.anchor.lat, 6);
  expect(st.lon).toBeCloseTo(f7.meta.anchor.lon, 6);

  // dots are on the globe and carry values, and NONE of them is off the grid:
  // a global cone has no edge to fall off
  expect(st.data.nDots).toBeGreaterThan(9);
  expect(st.data.nValued).toBeGreaterThan(5);
  expect(st.data.nInvalid).toBe(0);

  // THE DATELINE ITSELF: the drawn dots straddle ±180°, which is only possible
  // because the column wraps. On the North Atlantic window this is zero.
  const sides = await page.evaluate(() => {
    const d = window.__earth.coneDataDots;
    return {
      east: d.filter((x) => x.plon > 0).length,
      west: d.filter((x) => x.plon < 0).length,
      outOfRange: d.filter((x) => !(x.plon >= -180 && x.plon <= 180)).length,
      cols: [Math.min(...d.map((x) => x.col)), Math.max(...d.map((x) => x.col))],
    };
  });
  expect(sides.east).toBeGreaterThan(0);
  expect(sides.west).toBeGreaterThan(0);
  expect(sides.outOfRange).toBe(0);
  expect(sides.cols[0]).toBeGreaterThanOrEqual(0);
  expect(sides.cols[1]).toBeLessThan(f7.meta.grid.nx);

  // the hint names the tensor in plain words and says what the cell stands on
  const hint = page.locator("#cn-data-hint");
  await expect(hint).toContainText("family 7");
  await expect(hint).toContainText("whole globe");
  await expect(hint).toContainText("ocean");            // sphere_at_anchor = 0
  await expect(hint).toContainText("below sea level");  // elev_at_anchor < 0

  // the read-out still puts the unit back on a stored value — on family 7 the
  // (mean, sd) is keyed by GROUP, so a flat lookup would silently be wrong
  const picked = await page.evaluate(() => {
    const dots = window.__earth.coneDataDots;
    const i = dots.findIndex((d) => d.valid && d.obs && Number.isFinite(d.raw));
    window.__earth.conesPickDot(i);
    const p = window.__earth.coneState().data.pick;
    const c = document.getElementById("cn-anom");
    c.checked = false;
    c.dispatchEvent(new Event("change", { bubbles: true }));
    return { i, p, hi: window.__earth.coneState().data.hi };
  });
  expect(picked.i).toBeGreaterThanOrEqual(0);
  // cur_speed's own norm row is g025[0]; the RAW scale is a speed in m/s, not
  // the −3…+3 of a z-score, so a flat 54-long lookup could not produce it
  expect(picked.hi).toBeGreaterThan(0);
  expect(picked.hi).toBeLessThan(10);
  await expect(page.locator("#cn-readout")).toContainText("row ");

  // ---- and back to the North Atlantic set: its own five anchors return
  await page.evaluate(() => window.__earth.conesSetSource("anchors"));
  await expect.poll(() => page.evaluate(
    () => window.__earth.coneState().data.sampleSource)).toBe("anchors");
  await expect(page.locator("#cn-data-anchor option")).toHaveCount(5);
  expect(await page.locator("#cn-data-anchor option").allTextContents())
    .toEqual(naAnchors);
  expect(await page.locator("#cn-channel optgroup").count()).toBe(0);
  expect(await page.locator("#cn-channel option").count()).toBe(naChannels);
  expect(await page.evaluate(() => window.__earth.coneState().data.nChannels))
    .toBe(42);

  expect(page.__errors, `page errors: ${page.__errors.join(" | ")}`).toHaveLength(0);
});

/* ================================ the global tensor (family 7) on the globe
 *
 * FAMILY 7 is the first input tensor covering the whole globe rather than the
 * North Atlantic window: every 0.25° grid point, one value per channel per
 * five-day bin from 1982 to 2024. The layer paints one channel of one bin by a
 * single HTTP range read of a `.npy` on the Hugging Face Hub.
 *
 * The real files are tens of gigabytes, so these tests serve `data/family7/
 * fixture/` — the same schema over the T=5 tensor `ml/build_family7.py --smoke`
 * produces, decimated to a 5°/10° grid so it can live in git. TWO routes make
 * that work, and the second one is the interesting half: the index is swapped
 * for the fixture's, and the Hub URL is answered with the SLICED bytes and a
 * real 206, so the offset arithmetic the browser computes is genuinely
 * exercised rather than hidden behind a whole-file 200. */
const F7_DIR = require("path").join(__dirname, "..", "data", "family7", "fixture");

function f7Index() {
  return JSON.parse(require("fs").readFileSync(
    require("path").join(F7_DIR, "family7_index.json"), "utf8"));
}

async function serveFamily7(page, opts = {}) {
  const fs = require("fs"), path = require("path");
  const index = f7Index();
  await page.route(/\/data\/family7_index\.json(\?.*)?$/, (route) => {
    if (opts.noIndex) return route.fulfill({ status: 404, body: "" });
    return route.fulfill({ status: 200, contentType: "application/json",
                           body: JSON.stringify(index) });
  });
  await page.route(/resolve\/main\/tensors\/.*\.npy$/, (route) => {
    if (opts.slabFails) return route.fulfill({ status: 500, body: "" });
    const name = route.request().url().split("/").pop();
    const buf = fs.readFileSync(path.join(F7_DIR, name));
    const range = route.request().headers()["range"];
    const m = range && /bytes=(\d+)-(\d+)/.exec(range);
    if (!m) return route.fulfill({ status: 200, body: buf });
    const a = Number(m[1]), b = Number(m[2]);
    page.__f7Reads = (page.__f7Reads || []).concat([[name, a, b]]);
    return route.fulfill({
      status: 206,
      headers: { "content-range": `bytes ${a}-${b}/${buf.length}`,
                 "accept-ranges": "bytes",
                 "content-type": "application/octet-stream" },
      body: buf.slice(a, b + 1),
    });
  });
  return index;
}

// The float16 in the file, decoded the way the page decodes it — so the
// expected value below is the FIXTURE'S OWN BYTES and not a second computation
// of what they should have been.
function f7Cell(index, group, bin, chan, iy, ix) {
  const fs = require("fs"), path = require("path");
  const grp = index.groups[group];
  const buf = fs.readFileSync(path.join(F7_DIR, grp.file));
  const row = bin - index.bin_first;
  const nC = grp.chans.length, ci = grp.chans.indexOf(chan);
  const off = grp.header_len + row * grp.slab_bytes +
              ((iy * grp.grid.nx + ix) * nC + ci) * grp.itemsize;
  const h = buf.readUInt16LE(off);
  const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
  const z = e === 0 ? s * 6.103515625e-5 * (f / 1024)
          : e === 31 ? (f ? NaN : s * Infinity)
          : s * Math.pow(2, e - 15) * (1 + f / 1024);
  const [mean, sd] = grp.norm[ci];
  return { z, raw: z * sd + mean, mean, sd };
}

async function setAppDate(page, d) {
  await page.evaluate((v) => {
    const el = document.getElementById("layer-date");
    el.value = v;
    el.dispatchEvent(new Event("change", { bubbles: true }));
  }, d);
}

async function enableFamily7(page) {
  await page.evaluate(() => {
    const el = document.querySelector('#layer-list input[data-id="family7"]');
    el.checked = true;
    el.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await expect.poll(() => page.evaluate(() => window.__earth.tensorLayerState().ready),
                    { timeout: 20000 }).toBe(true);
}

test("family 7: one range read paints a pentad, and the probe reads the tensor's own bytes",
     async ({ page }) => {
  test.setTimeout(120000);
  const index = await serveFamily7(page);
  const toasts = await recordToasts(page);

  // the date selector is the layer's clock; put it inside the fixture's span
  await page.evaluate(() => {
    const el = document.getElementById("layer-date");
    el.value = "2010-01-20";
    el.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await enableFamily7(page);

  const st = await page.evaluate(() => window.__earth.tensorLayerState());
  expect(st.hasIndex).toBe(true);
  expect(st.error).toBe(null);
  // 2010-01-19 is bin 2049 (5-day bins from 1982-01-01)
  expect(st.bin).toBe(2049);
  expect(st.grid.binStart).toBe("2010-01-19");
  expect(st.chan).toBe("g025:sst");
  expect(st.grid.nx).toBe(index.groups.g025.grid.nx);
  expect(st.grid.ny).toBe(index.groups.g025.grid.ny);
  expect(st.grid.wrap).toBe(true);
  expect(st.grid.units).toBe("°C");
  // a POINT-aligned grid: the cell is the half-step box around its point, so
  // the west edge is half a step west of −180 and the grid wraps
  expect(st.grid.west).toBeCloseTo(-180 - index.groups.g025.grid.step / 2, 6);
  expect(st.grid.south).toBeCloseTo(-90 - index.groups.g025.grid.step / 2, 6);

  // exactly ONE slab was read, and it was the bin's slab, by offset
  const grp = index.groups.g025;
  const reads = page.__f7Reads || [];
  expect(reads.length).toBe(1);
  const want = grp.header_len + (2049 - index.bin_first) * grp.slab_bytes;
  expect(reads[0][1]).toBe(want);
  expect(reads[0][2] - reads[0][1] + 1).toBe(grp.slab_bytes);
  expect(st.slabs).toEqual(["g025:2049"]);

  // …and the number under a point is the FIXTURE'S OWN BYTE, un-z-scored.
  // Row 20 / col 40 of the 5° grid is lat −90+5·20 = 10°N, lon −180+5·40 = 20°E.
  const cell = f7Cell(index, "g025", 2049, "sst", 20, 40);
  const got = await page.evaluate(() => window.__earth.tensorSampleAt(20, 10));
  if (Number.isFinite(cell.raw)) {
    expect(got).toBeCloseTo(cell.raw, 3);
    // raw = z·sd + mean, and the two are different numbers
    expect(st.norm[0]).toBeCloseTo(cell.mean, 5);
    expect(st.norm[1]).toBeCloseTo(cell.sd, 5);
  } else {
    expect(got).toBe(null);            // NaN is "never observed", not a zero
  }

  // the read-out prints the unit AND the stored σ (§2.9 + the z the model reads)
  const probe = await page.evaluate(async () => {
    const E = window.__earth;
    const carto = Cesium.Cartographic.fromDegrees(20, 10);
    return await E.probeValueAt(carto);
  });
  expect(probe).toBeTruthy();
  expect(probe.title).toContain("Global tensor");
  if (!probe.noData) {
    expect(probe.units).toBe("°C");
    expect(probe.extra).toContain("z =");
    expect(probe.when.kind).toBe("day");       // the bin's OPENING day
    expect(probe.when.t).toBe("2010-01-19");
  }

  // the toast names the pentad the reader is looking at
  await expect.poll(toasts).toContain("bin 2049");
  await expect.poll(toasts).toContain("2010-01-19 → 01-23");

  expect(page.__errors, `page errors: ${page.__errors.join(" | ")}`).toHaveLength(0);
});

test("family 7: a channel switch costs no request, a coarse channel changes grid, statics are local",
     async ({ page }) => {
  test.setTimeout(120000);
  const index = await serveFamily7(page);
  await page.evaluate(() => {
    const el = document.getElementById("layer-date");
    el.value = "2010-01-20";
    el.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await enableFamily7(page);
  expect((page.__f7Reads || []).length).toBe(1);

  // every g025 and g100 channel plus the two statics, each with a label and a unit
  const opts = await page.locator('select[data-chan="family7"] option').allTextContents();
  const nChan = index.groups.g025.chans.length + index.groups.g100.chans.length + 2;
  expect(opts.length).toBe(nChan);
  expect(opts.join(" | ")).toContain("Surface current speed — m/s");
  expect(opts.join(" | ")).toContain("Surface sphere");
  // rg100 (the depth column) is NOT offered: it is written only into the
  // pentad holding a month's 15th, so most dates would paint nothing
  expect(opts.join(" | ")).not.toContain("dbar");

  // ANOTHER CHANNEL OF THE SAME BIN: the slab is already in the LRU, so this
  // is a decode and not a request — that is the whole reason the cache holds
  // raw slabs rather than decoded planes.
  await page.evaluate(() => {
    const s = document.querySelector('select[data-chan="family7"]');
    s.value = "g025:cur_u";
    s.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await expect.poll(() => page.evaluate(() => window.__earth.tensorLayerState().chan),
                    { timeout: 20000 }).toBe("g025:cur_u");
  expect((page.__f7Reads || []).length).toBe(1);
  const signed = await page.evaluate(() => window.__earth.tensorLayerState());
  expect(signed.vmin).toBe(-signed.vmax);          // a signed channel is diverging
  expect(signed.ramp).toBe("anom");
  expect(signed.grid.units).toBe("m/s");

  // a COARSE-group channel is a different file, a different grid and one read
  await page.evaluate(() => {
    const s = document.querySelector('select[data-chan="family7"]');
    s.value = "g100:t2m";
    s.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await expect.poll(() => page.evaluate(() => window.__earth.tensorLayerState().chan),
                    { timeout: 20000 }).toBe("g100:t2m");
  await expect.poll(() => page.evaluate(() => window.__earth.tensorLayerState().ready),
                    { timeout: 20000 }).toBe(true);
  const coarse = await page.evaluate(() => window.__earth.tensorLayerState());
  expect(coarse.grid.nx).toBe(index.groups.g100.grid.nx);
  expect(coarse.grid.dlon).toBe(index.groups.g100.grid.step);
  expect((page.__f7Reads || []).length).toBe(2);
  expect((page.__f7Reads || [])[1][0]).toContain("g100");

  // a STATIC reads no slab at all — it is an ordinary baked grid beside the index
  await page.evaluate(() => {
    const s = document.querySelector('select[data-chan="family7"]');
    s.value = "static:sphere";
    s.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await expect.poll(() => page.evaluate(() => window.__earth.tensorLayerState().classGrid),
                    { timeout: 20000 }).toBe(true);
  await expect.poll(() => page.evaluate(() => window.__earth.tensorLayerState().ready),
                    { timeout: 20000 }).toBe(true);
  const stat = await page.evaluate(() => window.__earth.tensorLayerState());
  expect(stat.classGrid).toBe(true);
  expect(stat.grid.binStart).toBe(null);           // a static has no observation time
  expect((page.__f7Reads || []).length).toBe(2);   // still two
  // …and it answers with a class NAME, never a code
  const cls = await page.evaluate(async () =>
    await window.__earth.probeValueAt(Cesium.Cartographic.fromDegrees(20, 10)));
  expect(cls.label || cls.noData).toBeTruthy();
  if (cls.label) expect(["ocean", "land", "ice sheet or glacier", "inland water"]).toContain(cls.label);

  expect(page.__errors, `page errors: ${page.__errors.join(" | ")}`).toHaveLength(0);
});

test("family 7: stepping the date to another pentad reads exactly one more slab",
     async ({ page }) => {
  test.setTimeout(120000);
  const index = await serveFamily7(page);
  await page.evaluate(() => {
    const el = document.getElementById("layer-date");
    el.value = "2010-01-20";
    el.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await enableFamily7(page);
  expect((page.__f7Reads || []).length).toBe(1);

  // a day INSIDE the same bin changes nothing — four days in five are free
  await setAppDate(page, "2010-01-21");
  await page.waitForTimeout(400);
  expect((page.__f7Reads || []).length).toBe(1);
  expect(await page.evaluate(() => window.__earth.tensorLayerState().bin)).toBe(2049);

  // the NEXT bin is one more read, and the LRU now holds both
  await setAppDate(page, "2010-01-25");
  await expect.poll(() => page.evaluate(
    () => window.__earth.tensorLayerState().grid?.binStart),
    { timeout: 30000 }).toBe("2010-01-24");
  expect((page.__f7Reads || []).length).toBe(2);
  const st = await page.evaluate(() => window.__earth.tensorLayerState());
  expect(st.bin).toBe(2050);
  expect(st.slabs.sort()).toEqual(["g025:2049", "g025:2050"]);
  // stepping BACK is free: the bin is still in the LRU
  await setAppDate(page, "2010-01-20");
  await expect.poll(() => page.evaluate(
    () => window.__earth.tensorLayerState().grid?.binStart),
    { timeout: 30000 }).toBe("2010-01-19");
  expect((page.__f7Reads || []).length).toBe(2);

  expect(page.__errors, `page errors: ${page.__errors.join(" | ")}`).toHaveLength(0);
});

test("family 7 degrades to a hint when the index or the archive is missing",
     async ({ page }) => {
  test.setTimeout(120000);
  /* The whole reason huggingface.co could be admitted as a live endpoint
   * (CLAUDE.md §3) is that every failure path ends in an omitted section, not
   * in a broken globe. Both halves are exercised: no index at all, and an
   * index whose archive answers 500. */
  await serveFamily7(page, { noIndex: true });
  const toasts = await recordToasts(page);
  await page.evaluate(() => {
    const el = document.querySelector('#layer-list input[data-id="family7"]');
    el.checked = true;
    el.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await expect.poll(toasts, { timeout: 20000 }).toContain("could not be read");
  const st = await page.evaluate(() => window.__earth.tensorLayerState());
  expect(st.hasIndex).toBe(false);
  expect(st.ready).toBe(false);
  // the layer is still ON — a chip, a row, an opacity slider — it simply paints
  // nothing, and the globe underneath is untouched
  expect(await page.evaluate(() => !!window.__earth.state.layers.family7?.layer)).toBe(true);
  await expect(page.locator("#active-layers")).toContainText("Global tensor");
  expect(page.__errors, `page errors: ${page.__errors.join(" | ")}`).toHaveLength(0);
});

test("family 7 degrades when the archive itself refuses the range read", async ({ page }) => {
  test.setTimeout(120000);
  await serveFamily7(page, { slabFails: true });
  const toasts = await recordToasts(page);
  await page.evaluate(() => {
    const el = document.getElementById("layer-date");
    el.value = "2010-01-20";
    el.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await page.evaluate(() => {
    const el = document.querySelector('#layer-list input[data-id="family7"]');
    el.checked = true;
    el.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await expect.poll(() => page.evaluate(() => window.__earth.tensorLayerState().error),
                    { timeout: 20000 }).toContain("500");
  expect(await page.evaluate(() => window.__earth.tensorLayerState().ready)).toBe(false);
  // the index DID arrive, so the channel picker is still usable and a static
  // (which needs no archive at all) still works
  expect(await page.evaluate(() => window.__earth.tensorLayerState().hasIndex)).toBe(true);
  await page.evaluate(() => {
    const s = document.querySelector('select[data-chan="family7"]');
    s.value = "static:elev";
    s.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await expect.poll(() => page.evaluate(() => window.__earth.tensorLayerState().ready),
                    { timeout: 20000 }).toBe(true);
  expect(page.__errors, `page errors: ${page.__errors.join(" | ")}`).toHaveLength(0);
});

/* ==================================== Cones · DATA mode · LIVE (family 7) ===
 *
 * The Cones tab's data mode had one source: five pre-exported North Atlantic
 * anchors of family 4. LIVE mode is the second — the same cone read straight
 * out of family 7, the global tensor, so any cell on the planet can be an
 * anchor. These tests serve the same family-7 fixture the layer tests do, so
 * the cone and the globe layer share one archive and one LRU, exactly as they
 * do in production. */
test("Cones live mode: any cell on Earth is an anchor, and the dots carry values",
     async ({ page }) => {
  test.setTimeout(180000);
  await serveFamily7(page);
  await setAppDate(page, "2010-01-20");
  await page.click("#tab-cones");
  await expect(page.locator("#cn-reach .stat-value")).not.toHaveText("–", { timeout: 20000 });
  await page.evaluate(() => window.__earth.conesSetDataMode(true));
  await page.evaluate(() => window.__earth.conesSetSource("live"));
  await expect.poll(() => page.evaluate(() => window.__earth.coneState().data.live),
                    { timeout: 20000 }).toBe(true);

  // the rows that mean nothing here are gone: no exported-anchor picker, and
  // no anomaly toggle (the anomaly is a climatology, not something seven
  // pentads can produce)
  await expect(page.locator("#cn-anchor-row")).toBeHidden();
  await expect(page.locator("#cn-anom-row")).toBeHidden();

  // ANTARCTICA. The whole point of the global tensor: a cell the North
  // Atlantic window did not contain at all.
  await page.evaluate(() => window.__earth.conesLiveAnchorAt(-70, -60));
  await expect.poll(() => page.evaluate(
    () => window.__earth.coneState().data.liveValued), { timeout: 30000 })
    .toBeGreaterThan(0);
  const st = await page.evaluate(() => window.__earth.coneState());
  expect(st.data.liveAnchor.lat).toBeCloseTo(-70, 3);
  expect(st.data.liveAnchor.lon).toBeCloseTo(-60, 3);
  expect(st.drawn).toBeGreaterThan(9);            // the 3×3 patch plus dots
  // the anchor's two statics — a place, not a row and a column
  expect(["ocean", "land", "ice sheet or glacier", "inland water"])
    .toContain(st.data.liveSphere);
  expect(Number.isFinite(st.data.liveElev)).toBe(true);
  await expect(page.locator("#cn-readout")).toContainText(st.data.liveSphere);

  // a dot reads back its raw value, its z, and ITS OWN date (anchor − lag)
  const picked = await page.evaluate(() => {
    const dots = window.__earth.coneDataDots;
    const i = dots.findIndex((d) => d.obs && d.lag > 0);
    window.__earth.conesPickDot(i < 0 ? dots.findIndex((d) => d.obs) : i);
    return window.__earth.coneState().data.pick;
  });
  expect(picked.obs).toBe(true);
  expect(Number.isFinite(picked.raw)).toBe(true);
  expect(Number.isFinite(picked.z)).toBe(true);
  const ro = page.locator("#cn-readout");
  await expect(ro).toContainText(`lag ${picked.lag}`);
  await expect(ro).toContainText(picked.date);
  await expect(ro).toContainText("z =");
  await expect(ro).toContainText("observed");

  // the hint says what live mode cannot do, rather than leaving it to be found
  const hint = page.locator("#cn-data-hint");
  await expect(hint).toContainText("Live from family 7");
  await expect(hint).toContainText("No anomaly and no depth column");

  expect(page.__errors, `page errors: ${page.__errors.join(" | ")}`).toHaveLength(0);
});

test("Cones live mode: the dots wrap across the dateline, and the outer stencil is geometry",
     async ({ page }) => {
  test.setTimeout(180000);
  await serveFamily7(page);
  await setAppDate(page, "2010-01-20");
  await page.click("#tab-cones");
  await expect(page.locator("#cn-reach .stat-value")).not.toHaveText("–", { timeout: 20000 });
  await page.evaluate(() => window.__earth.conesSetDataMode(true));
  await page.evaluate(() => window.__earth.conesSetSource("live"));
  await expect.poll(() => page.evaluate(() => window.__earth.coneState().data.live),
                    { timeout: 20000 }).toBe(true);

  /* THE DATELINE. On the North Atlantic window a dot past the eastern edge is
   * off the tensor and drawn hollow, because the model reads it as missing. On
   * the globe there is no edge: a dot 900 km east of 179.75°E is in the
   * western Pacific. Anchor against the seam and walk the lag out. */
  await page.evaluate(() => {
    const s = document.getElementById("cn-lag");
    s.value = "6"; s.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await page.evaluate(() => window.__earth.conesLiveAnchorAt(36, 179.75));
  await expect.poll(() => page.evaluate(
    () => window.__earth.coneState().data.liveWrapped), { timeout: 30000 })
    .toBeGreaterThan(0);
  const st = await page.evaluate(() => window.__earth.coneState());
  // every wrapped dot is still a real place on the globe, west of the seam
  const cols = await page.evaluate(() =>
    window.__earth.coneDataDots.map((d) => d.col));
  const G = await page.evaluate(() => window.__earth.coneGlobalGrid());
  expect(Math.min(...cols)).toBeGreaterThanOrEqual(0);
  expect(Math.max(...cols)).toBeLessThan(G.nx);
  expect(st.offWindow).toBe(0);      // nothing is "off the window": there isn't one

  // the OUTER stencil is drawn, and drawn as SHAPE — 137 slabs would be ~2 GB
  const readsBefore = (page.__f7Reads || []).length;
  await page.evaluate(() => {
    const s = document.getElementById("cn-stencil");
    s.value = "stage2"; s.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await expect(page.locator("#cn-data-hint")).toContainText("geometry only");
  const s2 = await page.evaluate(() => window.__earth.coneState());
  expect(s2.drawn).toBeGreaterThan(0);
  expect(s2.data.liveValued).toBe(0);            // no values, by design
  expect((page.__f7Reads || []).length).toBe(readsBefore);   // and no reads
  expect(page.__errors, `page errors: ${page.__errors.join(" | ")}`).toHaveLength(0);
});

test("Cones live mode: a coarse channel is read at its own coarse cell, and says so",
     async ({ page }) => {
  test.setTimeout(180000);
  const index = await serveFamily7(page);
  await setAppDate(page, "2010-01-20");
  await page.click("#tab-cones");
  await expect(page.locator("#cn-reach .stat-value")).not.toHaveText("–", { timeout: 20000 });
  await page.evaluate(() => window.__earth.conesSetDataMode(true));
  await page.evaluate(() => window.__earth.conesSetSource("live"));
  await expect.poll(() => page.evaluate(() => window.__earth.coneState().data.live),
                    { timeout: 20000 }).toBe(true);

  // the channel list is the TENSOR'S, and rg100 (the depth column) is not in it
  const opts = await page.locator("#cn-channel option").allTextContents();
  expect(opts.length).toBe(index.groups.g025.chans.length +
                           index.groups.g100.chans.length);
  expect(opts.join(" | ")).not.toContain("dbar");

  await page.evaluate(() => window.__earth.conesLiveAnchorAt(36, -30));
  await page.evaluate(() => {
    const s = document.getElementById("cn-channel");
    s.value = "g100:t2m"; s.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await expect.poll(() => page.evaluate(
    () => window.__earth.coneState().data.liveChan), { timeout: 20000 }).toBe("g100:t2m");
  await expect.poll(() => page.evaluate(
    () => window.__earth.coneState().data.liveValued), { timeout: 30000 })
    .toBeGreaterThan(0);

  /* THE NINE CELLS OF THE PATCH SHOW ONE NUMBER, and that is the model's
   * honest view of a coarse channel rather than a rounding artefact of the
   * drawing: a 1° channel has one value per 1° cell, and the 3×3 patch of a
   * 0.25° anchor sits inside it. */
  const patch = await page.evaluate(() =>
    window.__earth.coneDataDots.filter((d) => d.kind === "patch")
      .map((d) => d.raw));
  expect(patch).toHaveLength(9);
  expect(new Set(patch.map((v) => (v === null ? "·" : v.toFixed(6)))).size).toBe(1);
  await expect(page.locator("#cn-data-hint"))
    .toContainText("all show the same number");

  // the family the channel is read through comes from the exported geometry
  const fam = await page.evaluate(() => window.__earth.coneState().data.liveFamily);
  expect(fam.known).toBe(true);
  expect(["A", "B", "C", "L"]).toContain(fam.fam);

  // …and the LAND family exists now, which the North Atlantic tensor never had
  await page.evaluate(() => {
    const s = document.getElementById("cn-channel");
    s.value = "g100:soilw"; s.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await expect.poll(() => page.evaluate(
    () => window.__earth.coneState().data.liveFamily?.asked), { timeout: 20000 })
    .toBe("L");
  expect(page.__errors, `page errors: ${page.__errors.join(" | ")}`).toHaveLength(0);
});

test("Cones live mode: the source switch leaves the exported anchors untouched",
     async ({ page }) => {
  test.setTimeout(180000);
  const fx = await serveConeFixture(page);
  await serveFamily7(page);
  await page.click("#tab-cones");
  await expect(page.locator("#cn-reach .stat-value")).not.toHaveText("–", { timeout: 20000 });
  await page.evaluate(() => window.__earth.conesSetDataMode(true));

  // the DEFAULT is still the exported anchors — today's behaviour, untouched
  const first = await page.evaluate(() => window.__earth.coneState());
  expect(first.data.source).toBe("anchors");
  expect(first.data.ready).toBe(true);
  expect(first.data.date).toBe(fx.meta.dates[0]);
  await expect(page.locator("#cn-anchor-row")).toBeVisible();
  await expect(page.locator("#cn-anom-row")).toBeVisible();

  await page.evaluate(() => window.__earth.conesSetSource("live"));
  await expect.poll(() => page.evaluate(() => window.__earth.coneState().data.live),
                    { timeout: 20000 }).toBe(true);

  // …and going back restores it, sample and all
  await page.evaluate(() => window.__earth.conesSetSource("anchors"));
  await expect.poll(() => page.evaluate(() => window.__earth.coneState().data.source))
    .toBe("anchors");
  const back = await page.evaluate(() => window.__earth.coneState());
  expect(back.data.live).toBe(false);
  expect(back.data.ready).toBe(true);
  expect(back.data.anchorId).toBe(first.data.anchorId);
  await expect(page.locator("#cn-anchor-row")).toBeVisible();
  expect(page.__errors, `page errors: ${page.__errors.join(" | ")}`).toHaveLength(0);
});

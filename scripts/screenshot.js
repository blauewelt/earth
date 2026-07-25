#!/usr/bin/env node
// Regenerates docs/screenshot.png (the README hero image).
//
// Run it the same way the tests run — through the sandbox proxies, so the
// globe gets real GIBS tiles even where the browser has no direct egress:
//
//   scripts/run_tests.sh  ... is for tests; for the screenshot use:
//   python3 scripts/test_proxy.py 8081 https://gibs.earthdata.nasa.gov &
//   python3 scripts/test_proxy.py 8082 https://api.gbif.org &
//   python3 -m http.server 8080 &
//   MIRROR=1 node scripts/screenshot.js
//
// Without MIRROR it just points at http://localhost:8080 and uses the real
// network, which is what you want on a normal machine.
"use strict";
const { chromium } = require("@playwright/test");
const path = require("path");

const BASE = process.env.BASE_URL || "http://localhost:8080";
const CDN = "https://cdnjs.cloudflare.com/ajax/libs/cesium/1.133.1";
const OUT = path.join(__dirname, "..", "docs", "screenshot.png");

(async () => {
  const browser = await chromium.launch({
    executablePath: process.env.CHROMIUM_PATH || undefined,
  });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });

  if (process.env.MIRROR) {
    const proxy = (re, from, to) =>
      page.route(re, async (route) => {
        try {
          const url = route.request().url().replace(from, to)
            .replace("widgets.min.css", "widgets.css");
          await route.fulfill({ response: await page.request.get(url) });
        } catch {
          await route.abort().catch(() => {});
        }
      });
    await proxy(/https:\/\/cdnjs\.cloudflare\.com\/.*/, CDN, `${BASE}/_vendor/cesium`);
    await proxy(/https:\/\/gibs\.earthdata\.nasa\.gov\/.*/,
      "https://gibs.earthdata.nasa.gov", "http://localhost:8081");
    await proxy(/https:\/\/api\.gbif\.org\/.*/, "https://api.gbif.org", "http://localhost:8082");
  }

  await page.goto(BASE);
  await page.waitForFunction(() => window.__earth?.viewer, null, { timeout: 60000 });

  // A representative state rather than the bare default: SST and the station
  // network are on by default, Climate TRACE adds a third chip, the split
  // comparison shows the swipe divider and its date labels sitting below the
  // chip bar, and the AMOC dashboard fills the sidebar with real content.
  await page.check("#toggle-climatetrace");
  await page.selectOption("#compare-select", "10");
  await page.selectOption("#compare-mode", "split");
  await page.click("#tab-amoc");

  // Let the tiles for both sides of the divider settle.
  await page.waitForTimeout(15000);
  await page.screenshot({ path: OUT });
  await browser.close();
  console.log(`wrote ${OUT}`);
})();

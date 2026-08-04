const { test, expect } = require("@playwright/test");

test.use({ viewport: { width: 412, height: 915 }, hasTouch: true });

test("phone: tap with card open rotates mark out from under it", async ({ page }) => {
  page.__errors = [];
  page.on("pageerror", (e) => page.__errors.push(String(e)));
  if (process.env.MIRROR) {
    await page.route(/https:\/\/gibs\.earthdata\.nasa\.gov\/.*/, (route) =>
      route.continue({ url: route.request().url().replace("https://gibs.earthdata.nasa.gov", "http://localhost:8081") }));
    await page.route(/https:\/\/api\.gbif\.org\/.*/, (route) =>
      route.continue({ url: route.request().url().replace("https://api.gbif.org", "http://localhost:8082") }));
    await page.route(/https:\/\/api\.open-meteo\.com\/.*/, (route) =>
      route.continue({ url: route.request().url().replace("https://api.open-meteo.com", "http://localhost:8083") }));
    await page.route(/https:\/\/air-quality-api\.open-meteo\.com\/.*/, (route) =>
      route.continue({ url: route.request().url().replace("https://air-quality-api.open-meteo.com", "http://localhost:8084") }));
    await page.route(/https:\/\/flood-api\.open-meteo\.com\/.*/, (route) =>
      route.continue({ url: route.request().url().replace("https://flood-api.open-meteo.com", "http://localhost:8085") }));
    await page.route(/https:\/\/marine-api\.open-meteo\.com\/.*/, (route) =>
      route.continue({ url: route.request().url().replace("https://marine-api.open-meteo.com", "http://localhost:8086") }));
    await page.route(/https:\/\/climate-api\.open-meteo\.com\/.*/, (route) =>
      route.continue({ url: route.request().url().replace("https://climate-api.open-meteo.com", "http://localhost:8087") }));
  }
  await page.goto("http://localhost:8080/");
  await page.waitForFunction(() => window.__earth?.viewer);
  await page.waitForTimeout(2500);

  const r = await page.evaluate(async () => {
    const E = window.__earth;
    const scene = E.viewer.scene;
    const canvas = scene.canvas;
    const out = { canvasW: canvas.clientWidth, canvasH: canvas.clientHeight };
    // zoom like the user's screenshot: globe fills the map area
    E.viewer.camera.setView({ destination: Cesium.Cartesian3.fromDegrees(10, 50, 3.0e6) });
    const pick = (x, y) =>
      E.viewer.camera.pickEllipsoid(new Cesium.Cartesian2(x, y), scene.globe.ellipsoid);
    // tap in the middle of where the pixel card will sit
    const tapX = canvas.clientWidth - 160, tapY = 200;
    const cart = pick(tapX, tapY);
    out.picked = !!cart;
    if (!cart) return out;
    const camBefore = E.viewer.camera.positionCartographic.clone();
    E.showPixelState(Cesium.Cartographic.fromCartesian(cart)).catch(() => {});
    await new Promise((res) => setTimeout(res, 2000));
    const camAfter = E.viewer.camera.positionCartographic;
    out.moved = Math.abs(camAfter.longitude - camBefore.longitude) +
                Math.abs(camAfter.latitude - camBefore.latitude) > 1e-4;
    const card = document.getElementById("pixel-card").getBoundingClientRect();
    out.card = { l: card.left, r: card.right, t: card.top, b: card.bottom, w: card.width };
    const cr = canvas.getBoundingClientRect();
    out.crect = { l: cr.left, t: cr.top, w: cr.width, h: cr.height };
    const st = Cesium.SceneTransforms;
    const toWin = (st.worldToWindowCoordinates || st.wgs84ToWindowCoordinates).bind(st);
    const m = E.probeMark;
    out.dotShown = m?.dot.show;
    const w = m && toWin(scene, m.dot.position.getValue(E.viewer.clock.currentTime));
    out.markWin = w ? { x: w.x, y: w.y } : null;
    out.underCard = !!w &&
      w.x + cr.left >= card.left && w.x + cr.left <= card.right &&
      w.y + cr.top >= card.top && w.y + cr.top <= card.bottom;
    // would ensureMarkVisible find candidates? probe its internals indirectly
    out.movesAgain = E.ensureMarkVisible();
    return out;
  });
  console.log("REPRO:", JSON.stringify(r, null, 1));
  expect(r.picked).toBe(true);
});

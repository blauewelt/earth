// Browser tests for docs.html — the mobile markdown reader.
//
// The page exists because the project's documents ARE the project and Chris
// reads them on a phone: ml/EXPERIMENTS.md is 2,400 lines with nine-column
// result tables. So the assertions here are about the two things that make
// those documents readable on a 360px screen, and both are behaviours a
// careless CSS edit can silently destroy:
//
//   1. NOTHING widens the page. A single un-contained table or code block
//      turns the whole document into a pinch-and-pan experience, and the
//      symptom (body text running off-screen) looks like a font problem.
//   2. The row label STAYS while the numbers scroll. That is the entire
//      value of the wide-table treatment — scrolling right to read 0.5883
//      is useless if you can no longer see which stencil it belongs to —
//      and it is asserted by actually scrolling the container and
//      re-measuring, not by checking that a class was applied.
//
// No network: docs.html reads the working copy when served from localhost,
// so these run against the REAL documents in the repo. That is deliberate —
// a doc that grows a table the renderer mishandles should fail the suite.
"use strict";
const { test, expect } = require("@playwright/test");

const PHONE = { width: 360, height: 800 };

test.describe("docs.html · mobile markdown reader", () => {
  test.use({ viewport: PHONE });

  test("index lists the document set", async ({ page }) => {
    await page.goto("/docs.html");
    await expect(page.locator(".idx a").first()).toBeVisible();
    const links = await page.locator(".idx a").count();
    expect(links).toBeGreaterThan(10);
    // every entry points at a markdown file inside the repo
    const hrefs = await page.locator(".idx a").evaluateAll((as) =>
      as.map((a) => a.getAttribute("href")));
    for (const h of hrefs) expect(h).toMatch(/^\?f=[\w./%-]+\.md$/i);
  });

  /* A plan nobody can open is a plan nobody reads. CLAUDE.md SS0b says a new
   * markdown document gets one line in docs.html's DOCS — and that step was
   * missed twice in one day (E-038 and E-039, 2026-08-16), each time silently:
   * the file was on main, the blob URL was correct, and the reader's index
   * simply did not know it existed. Nothing in the suite noticed, because
   * every test asked about documents that WERE listed. So ask the filesystem
   * instead of the list: ml/plans/ is the directory whose whole purpose is to
   * be read by someone deciding whether to spend a week, and it is small
   * enough that "all of them" is the right rule. */
  test("every experiment plan is reachable from the docs reader", async ({ page }) => {
    const fs = require("fs"), path = require("path");
    const dir = path.join(__dirname, "..", "ml", "plans");
    const plans = fs.readdirSync(dir).filter((f) => f.endsWith(".md")).sort();
    expect(plans.length).toBeGreaterThan(0);
    await page.goto("/docs.html");
    const hrefs = await page.locator(".idx a").evaluateAll((as) =>
      as.map((a) => decodeURIComponent(a.getAttribute("href") || "")));
    const listed = new Set(hrefs.map((h) => h.replace(/^\?f=/, "")));
    const missing = plans.filter((f) => !listed.has("ml/plans/" + f));
    expect(missing, `add these to DOCS in docs.html: ${missing.join(", ")}`)
      .toEqual([]);
  });

  // The whole point. Checked on the biggest, widest document we have.
  test("no document widens the page", async ({ page }) => {
    for (const doc of ["ml/EXPERIMENTS.md", "ml/LEADERBOARD.md", "CLAUDE.md",
                       "docs/PIXEL_STATE.md"]) {
      await page.goto("/docs.html?f=" + doc);
      await expect(page.locator("main h1, main h2").first()).toBeVisible();
      const width = await page.evaluate(() => document.documentElement.scrollWidth);
      expect(width, doc + " overflows the viewport").toBeLessThanOrEqual(PHONE.width + 1);
    }
  });

  test("a wide table keeps its row label pinned while the numbers scroll",
    async ({ page }) => {
      await page.goto("/docs.html?f=ml/EXPERIMENTS.md");
      await expect(page.locator("table.pin").first()).toBeVisible();

      const r = await page.evaluate(() => {
        // any table wide enough to actually overflow its box
        const box = [...document.querySelectorAll(".tbox.over")][0];
        if (!box) return null;
        const sc = box.querySelector(".tscroll");
        const t = box.querySelector("table");
        // the label column is the first, or the second when the first is a rank
        const i = t.classList.contains("pin2") ? 1 : 0;
        const cell = box.querySelector("tbody tr").cells[i];
        const before = cell.getBoundingClientRect().left;
        const label = cell.textContent.trim();
        sc.scrollLeft = sc.scrollWidth;                 // swipe to the far end
        return { label, before, after: cell.getBoundingClientRect().left,
                 moved: sc.scrollLeft };
      });

      expect(r, "no overflowing table found to test").not.toBeNull();
      expect(r.moved).toBeGreaterThan(20);              // it really did scroll
      expect(r.label.length).toBeGreaterThan(0);        // and it says something
      expect(Math.abs(r.after - r.before)).toBeLessThan(2);   // and it stayed put
    });

  test("a two-column prose table becomes labelled blocks", async ({ page }) => {
    await page.goto("/docs.html?f=CLAUDE.md");
    const stacked = page.locator("table.stack").first();
    await expect(stacked).toBeVisible();
    // each cell carries its column name, which is what replaces the header row
    const labels = await stacked.locator("tbody td").evaluateAll((tds) =>
      tds.map((td) => td.getAttribute("data-label")).filter(Boolean));
    expect(labels.length).toBeGreaterThan(0);
    // and the header row is gone at this width, not merely visually empty
    await expect(stacked.locator("thead")).toBeHidden();
  });

  test("a long document gets a working contents drawer", async ({ page }) => {
    await page.goto("/docs.html?f=ml/EXPERIMENTS.md");
    await expect(page.locator("main h2").first()).toBeVisible();
    await expect(page.locator("#tocBtn")).toBeVisible();
    expect(await page.locator("#tocNav a").count()).toBeGreaterThan(10);

    await page.click("#tocBtn");
    await expect(page.locator("#toc")).toHaveClass(/on/);
    // every entry resolves to a heading that exists
    const missing = await page.evaluate(() =>
      [...document.querySelectorAll("#tocNav a")]
        .filter((a) => !document.getElementById(a.getAttribute("href").slice(1)))
        .length);
    expect(missing).toBe(0);

    await page.click("#tocNav a:nth-child(2)");
    await expect(page.locator("#toc")).not.toHaveClass(/on/);   // closes on pick
  });

  test("only markdown inside the repo is ever loaded", async ({ page }) => {
    for (const bad of ["https://example.com/x.md", "../../etc/passwd",
                       "src/app.js"]) {
      await page.goto("/docs.html?f=" + encodeURIComponent(bad));
      // falls back to the index rather than fetching it
      await expect(page.locator(".idx a").first()).toBeVisible();
    }
  });
});

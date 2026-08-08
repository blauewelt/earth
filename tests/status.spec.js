// Browser tests for status.html — the ML status page.
//
// This page was the one deployed artifact in the repo with no test at all,
// and it bit us: an edit once left `cfg` read but never declared, which
// `node --check` accepts and "use strict" turns into a ReferenceError that
// blanks the whole training-curves section. Here the page is loaded for real
// with every GitHub endpoint stubbed by `page.route`, so no network is
// touched and the fixtures are exact — which is what makes it possible to
// assert on things like "the parent run's step-0 probe is on the chart".
//
// Nothing here depends on MIRROR: the page's only hosts are api.github.com
// and raw.githubusercontent.com, and both are intercepted below.
"use strict";
const { test, expect } = require("@playwright/test");

const NOW = Date.now();
const iso = (minsAgo) => new Date(NOW - minsAgo * 60000).toISOString();

function jsonl(rows) {
  return rows.map((r) => JSON.stringify(r)).join("\n");
}

// A parent job: trained 0 → 30,000, cancelled. Carries the step-0 probe,
// which is the untrained-codec reference the probe chart is read against.
const RUN_A = jsonl([
  { config: { steps: 60000, batch: 256, d_z: 64, params_M: 40.7, data: "f3_na025.npz", C: 39, T: 528 } },
  { step: 0, linear_r_deseas: 0.516, wall_s: 11 },
  { step: 300, loss_rec: 1.00, loss_nei: 2.00 },
  { step: 10000, loss_rec: 0.80, loss_nei: 1.80 },
  { step: 10000, linear_r_deseas: 0.550, wall_s: 3000 },
  { step: 22500, loss_rec: 0.70, loss_nei: 1.70 },
  { step: 22500, linear_r_deseas: 0.572, wall_s: 6800 },
  // Points PAST the checkpoint the child resumed from: they belong to a
  // trajectory that was thrown away and must not appear on the child's chart.
  { step: 30000, loss_rec: 0.60, loss_nei: 1.60 },
]);

// The continuation: same config, resumed at 22,500, still running.
const RUN_B = jsonl([
  { config: { steps: 60000, batch: 256, d_z: 64, params_M: 40.7, data: "f3_na025.npz", C: 39, T: 528, resume: "run-101" } },
  { resumed: { from: "run-101.pt", parent_tag: "run-101", at_step: 22500 } },
  { step: 22650, loss_rec: 0.690, loss_nei: 1.690 },
  { step: 35000, loss_rec: 0.660, loss_nei: 1.660 },
  { step: 35000, linear_r_deseas: 0.601, wall_s: 3200 },
]);

const RUNS = {
  workflow_runs: [
    {
      id: 2, run_number: 102, status: "in_progress", conclusion: null,
      created_at: iso(90), html_url: "https://example.invalid/102",
      display_title: "f3_anchor41M (continued)", name: "ml-train",
      head_sha: "b".repeat(40),
    },
    {
      id: 1, run_number: 101, status: "completed", conclusion: "cancelled",
      created_at: iso(300), html_url: "https://example.invalid/101",
      display_title: "f3_anchor41M", name: "ml-train",
      head_sha: "a".repeat(40),
    },
  ],
};

const DOCS = {
  101: "f3_anchor41M: the anchored 40.7M codec on the 0.25-degree tensor.",
  102: "f3_anchor41M continued from #101 with the GPU-probe fix.",
};

test.beforeEach(async ({ page }) => {
  await page.route(/https:\/\/api\.github\.com\/.*/, async (route) => {
    const url = route.request().url();
    let body = {};
    if (/\/actions\/workflows\/[^/]+\/runs/.test(url)) body = RUNS;
    else if (/\/releases/.test(url)) body = [];
    await route.fulfill({
      status: 200, contentType: "application/json", body: JSON.stringify(body),
    });
  });
  await page.route(/https:\/\/raw\.githubusercontent\.com\/.*/, async (route) => {
    const url = route.request().url();
    const fulfill = (b) => route.fulfill({ status: 200, contentType: "text/plain", body: b });
    if (/run_docs\.json/.test(url)) return fulfill(JSON.stringify(DOCS));
    // The live branch exists only for the running job; the finished one is
    // served from the ml-metrics archive. Model both, since the stitch has
    // to reach the ARCHIVE to find the parent.
    if (/ml-live-102\/metrics\.jsonl/.test(url)) return fulfill(RUN_B);
    if (/ml-metrics\/run-101\.jsonl/.test(url)) return fulfill(RUN_A);
    return route.fulfill({ status: 404, body: "" });
  });
});

test("status page renders training curves without a script error", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/status.html");
  await expect(page.locator("#live .card").first()).toBeVisible();
  expect(errors).toEqual([]);
  // Two charts per card: loss and probe.
  await expect(page.locator("#live .card").first().locator("svg.chart")).toHaveCount(2);
});

test("a resumed run charts the WHOLE trajectory, parent included", async ({ page }) => {
  await page.goto("/status.html");
  const card = page.locator("#live .card").first();
  await expect(card).toContainText("run #102");

  // The card says where the trajectory began and where the seam is.
  await expect(card).toContainText("continues run #101 from step 22,500");

  // The parent's step-0 probe — the untrained-codec reference — is back.
  // Without the stitch this line cannot exist: the child never measured it.
  await expect(card).toContainText("untrained codec 0.516");
  await expect(card).toContainText("+0.085 vs the untrained codec");

  // The seam is DRAWN, on both charts, rather than the join being hidden.
  const seams = await card.locator('svg.chart line[stroke="#d2a8ff"]').count();
  expect(seams).toBe(2);

  // The loss polyline spans the whole run: 5 points (parent 300/10,000/22,500
  // + child 22,650/35,000), NOT the child's 2, and NOT 6 — the parent's
  // step-30,000 point is past the resume and belongs to a discarded branch.
  const pts = await card.locator("svg.chart polyline").first().getAttribute("points");
  expect(pts.trim().split(/\s+/).length).toBe(5);

  // The x-axis is framed on the planned 60,000 steps, so the curve visibly
  // has a third of the run left.
  await expect(card.locator("svg.chart").first()).toContainText("60,000");
});

test("ETA is measured against this job's own steps, not the inherited ones", async ({ page }) => {
  await page.goto("/status.html");
  const card = page.locator("#live .card").first();
  // wall_s 3,200 over the 12,500 steps THIS job did = 0.256 s/step; 25,000
  // steps remain → ~1.8 h. Charging the parent's 22,500 steps to this job's
  // clock would report ~0.6 h — a number the box would blow straight past.
  await expect(card).toContainText("~1.8 h left");
});

test("a run with no parent is charted exactly as it always was", async ({ page }) => {
  await page.goto("/status.html");
  // #101 is completed and its metrics come from the archive; it has no
  // resume record, so nothing is stitched and no seam is drawn.
  // Match on the card's own HEADING, not its text: #102's card mentions
  // "run #101" in its continuation note, so a text filter finds both.
  const card = page.locator("#live .card")
    .filter({ has: page.locator("h3", { hasText: "run #101" }) });
  await expect(card).toHaveCount(1);
  await expect(card).not.toContainText("continues run");
  expect(await card.locator('svg.chart line[stroke="#d2a8ff"]').count()).toBe(0);
  // It keeps its own step-0 point, which is where the reference came from.
  await expect(card).toContainText("untrained codec 0.516");
});

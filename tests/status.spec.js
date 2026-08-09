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
// A stage-2 run: the temporal transformer's own curve and verdict, written
// into the SAME metrics.jsonl by temporal.py.
const RUN_S2 = jsonl([
  { config: { steps: 60000, params_M: 40.7, data: "f3_na025.npz", C: 39 } },
  { step: 300, loss_rec: 0.09, loss_nei: 0.10 },
  { step: 60000, loss_rec: 0.09, loss_nei: 0.10 },
  { stage2_config: { d_model: 320, layers: 6, K: 24, steps: 6000, params_M: 7.469, d_z: 64, seed: 0, tag: "" } },
  { stage2_step: 1500, stage2_zmse: 0.910, stage2_wall_s: 40 },
  { stage2_step: 3000, stage2_zmse: 0.845, stage2_wall_s: 80 },
  { stage2_step: 4500, stage2_zmse: 0.802, stage2_wall_s: 120 },
  { stage2_step: 6000, stage2_zmse: 0.781, stage2_wall_s: 160 },
  { stage2_result: { d_model: 320, layers: 6, K: 24, steps: 6000, params_M: 7.469,
                     seed: 0, tag: "",
                     z_mse_model: 0.731, z_mse_persistence: 1.173,
                     chan_mse_model: 0.728, chan_mse_persistence: 1.141,
                     rapid_r_deseas: 0.385, rapid_r_raw: 0.341 } },
]);

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
      id: 3, run_number: 103, status: "completed", conclusion: "success",
      created_at: iso(30), html_url: "https://example.invalid/103",
      display_title: "STAGE 2 xxlarge (320x6)", name: "ml-train",
      head_sha: "c".repeat(40),
    },
    {
      id: 5, run_number: 105, status: "queued", conclusion: null,
      created_at: iso(94), html_url: "https://example.invalid/105",
      display_title: "queued stage-2 job", name: "ml-train",
      head_sha: "e".repeat(40),
    },
    {
      id: 4, run_number: 104, status: "in_progress", conclusion: null,
      created_at: iso(40), run_started_at: iso(12),
      html_url: "https://example.invalid/104",
      display_title: "f3 build", name: "ml-train",
      head_sha: "d".repeat(40),
    },
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
    if (/ml-metrics\/fleet\.json/.test(url)) {
      // $20 credit, $1/h burn, snapshot 2 h old -> $18 left, 18 h runway.
      return fulfill(JSON.stringify({
        at: new Date(NOW - 2 * 3600000).toISOString(),
        credit_usd: 20, balance_usd: 0,
        boxes_total: 3, boxes_running: 2, burn_usd_per_h: 1.0,
        boxes: [{ id: 1, status: "running", dph: 0.5, disk_used_gb: 28, disk_gb: 50 },
                { id: 2, status: "running", dph: 0.5, disk_used_gb: 41, disk_gb: 50 }],
      }));
    }
    // The live branch exists only for the running job; the finished one is
    // served from the ml-metrics archive. Model both, since the stitch has
    // to reach the ARCHIVE to find the parent.
    if (/ml-live-102\/metrics\.jsonl/.test(url)) return fulfill(RUN_B);
    if (/ml-metrics\/run-103\.jsonl/.test(url)) return fulfill(RUN_S2);
    if (/ml-metrics\/run-101\.jsonl/.test(url)) return fulfill(RUN_A);
    // #104 is mid-build: it has a phase file and no metrics yet.
    if (/ml-live-104\/phase\.json/.test(url)) {
      return fulfill(JSON.stringify({
        phase: "building the tensor",
        detail: "assembling channels into the pixel tensor",
        at: new Date(NOW - 6 * 60000).toISOString(),
      }));
    }
    return route.fulfill({ status: 404, body: "" });
  });
});

test("status page renders training curves without a script error", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/status.html");
  await expect(page.locator("#live .card").first()).toBeVisible();
  expect(errors).toEqual([]);
  // Two charts on a TRAINING run's card: loss and probe. Selected by heading,
  // not .first() — the list is ordered in-progress-then-finished and gains
  // cards as runs are added, so position is not identity.
  const trained = page.locator("#live .card")
    .filter({ has: page.locator("h3", { hasText: "run #102" }) });
  await expect(trained.locator("svg.chart")).toHaveCount(2);
});

test("a resumed run charts the WHOLE trajectory, parent included", async ({ page }) => {
  await page.goto("/status.html");
  const card = page.locator("#live .card")
    .filter({ has: page.locator("h3", { hasText: "run #102" }) });
  await expect(card).toHaveCount(1);

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
  const card = page.locator("#live .card")
    .filter({ has: page.locator("h3", { hasText: "run #102" }) });
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


test("a stage-2 run gets its own chart and verdict", async ({ page }) => {
  await page.goto("/status.html");
  const card = page.locator("#live .card")
    .filter({ has: page.locator("h3", { hasText: "run #103" }) });
  await expect(card).toHaveCount(1);

  // Its own curve, in its own colour, with the config stated.
  await expect(card).toContainText("temporal transformer over the frozen embeddings");
  await expect(card).toContainText("320\u00d76");
  await expect(card).toContainText("7.469M params");
  expect(await card.locator('svg.chart polyline[stroke="#a371f7"]').count()).toBe(1);

  // The z-space curve is charted against the persistence bar it must beat.
  await expect(card).toContainText("persistence 1.173");
  await expect(card).toContainText("latest 0.7810");

  // The verdict is expressed as % better than persistence — what stage 2
  // actually optimises — with the noisy RAPID read-out clearly secondary.
  await expect(card).toContainText("z-space 37.7% better than persistence");
  await expect(card).toContainText("channel-space 36.2%");
  await expect(card).toContainText("RAPID r +0.385");
});

test("runs without stage 2 show no stage-2 panel", async ({ page }) => {
  await page.goto("/status.html");
  const card = page.locator("#live .card")
    .filter({ has: page.locator("h3", { hasText: "run #101" }) });
  await expect(card).not.toContainText("temporal transformer over the frozen");
  expect(await card.locator('svg.chart polyline[stroke="#a371f7"]').count()).toBe(0);
});


test("a queued run says it is queued, not building", async ({ page }) => {
  await page.goto("/status.html");
  const card = page.locator("#live .card")
    .filter({ has: page.locator("h3", { hasText: "run #105" }) });
  await expect(card).toHaveCount(1);
  await expect(card).toContainText("queued — waiting for a free runner");
  await expect(card).toContainText("Nothing is building yet");
  // The old page said this about every run without curves. It must not.
  await expect(card).not.toContainText("building dataset / seeding cache");
});

test("a running run names the phase it is actually in", async ({ page }) => {
  await page.goto("/status.html");
  const card = page.locator("#live .card")
    .filter({ has: page.locator("h3", { hasText: "run #104" }) });
  await expect(card).toHaveCount(1);
  await expect(card).toContainText("building the tensor");
  await expect(card).toContainText("assembling channels into the pixel tensor");
  await expect(card).toContainText("since 6 min ago");
});


test("a queued run reports waiting, never 'started'", async ({ page }) => {
  await page.goto("/status.html");
  const card = page.locator("#live .card")
    .filter({ has: page.locator("h3", { hasText: "run #105" }) });
  // Dispatched 94 min ago and never picked up. "queued, started 1 h 34 min
  // ago" is a contradiction: nothing has started.
  await expect(card.locator("h3")).toContainText("queued 1 h 34 min ago");
  await expect(card.locator("h3")).toContainText("not started yet");
  await expect(card.locator("h3")).not.toContainText("started 1 h 34 min ago");
});

test("a running run times from when a runner picked it up", async ({ page }) => {
  await page.goto("/status.html");
  const card = page.locator("#live .card")
    .filter({ has: page.locator("h3", { hasText: "run #104" }) });
  // Dispatched 40 min ago, started 12 — the elapsed time that matters is 12.
  await expect(card.locator("h3")).toContainText("started 12 min ago");
  await expect(card.locator("h3")).not.toContainText("40 min ago");
});


test("the fleet panel projects credit forward from the burn rate", async ({ page }) => {
  await page.goto("/status.html");
  const fleet = page.locator("#fleet");
  // $20 snapshot, $1/h, taken 2 h ago -> $18 left and 18 h of runway. The
  // projection is the point: a snapshot alone goes stale and quietly misleads.
  await expect(fleet).toContainText("$18.00");
  await expect(fleet).toContainText("credit left (projected)");
  await expect(fleet).toContainText("burn $1.000/h");
  await expect(fleet).toContainText("2 of 3 boxes running");
  await expect(fleet).toContainText("runway");
  // and it says how old the snapshot is, so stale reads as stale
  await expect(fleet).toContainText("2 h 0 min ago");
  // disk is shown because a full disk takes a runner offline
  await expect(fleet).toContainText("28/50 GB");
});

test("the fleet panel warns when runway is short", async ({ page }) => {
  await page.route(/ml-metrics\/fleet\.json/, (route) =>
    route.fulfill({ status: 200, contentType: "text/plain", body: JSON.stringify({
      at: new Date(NOW).toISOString(), credit_usd: 3, balance_usd: 0,
      boxes_total: 3, boxes_running: 3, burn_usd_per_h: 0.9, boxes: [],
    }) }));
  await page.goto("/status.html");
  // 3 / 0.9 = 3.3 h — well under the 12 h threshold.
  await expect(page.locator("#fleet")).toContainText("under 12 h of runway");
});

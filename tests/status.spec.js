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
const path = require("path");
const { pathToFileURL } = require("url");

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
  { stage2_config: { d_model: 320, layers: 6, K: 24, steps: 6000, params_M: 7.469,
                     batch: 512, train_windows: 26073420, d_z: 64, seed: 0, tag: "" } },
  { stage2_step: 1500, stage2_zmse: 0.910, stage2_wall_s: 40 },
  { stage2_step: 3000, stage2_zmse: 0.845, stage2_wall_s: 80 },
  { stage2_step: 4500, stage2_zmse: 0.802, stage2_wall_s: 120 },
  { stage2_step: 6000, stage2_zmse: 0.781, stage2_wall_s: 160,
    stage2_val_zmse: 0.856, stage2_amp: 0.72, stage2_grad_norm: 1.31 },
  { stage2_probe: { step: 3000, rapid_r_deseas: 0.21 } },
  { stage2_probe: { step: 6000, rapid_r_deseas: 0.34 } },
  { stage2_result: { d_model: 320, layers: 6, K: 24, steps: 6000, params_M: 7.469,
                     seed: 0, tag: "",
                     z_mse_model: 0.731, z_mse_persistence: 1.173,
                     chan_mse_model: 0.728, chan_mse_persistence: 1.141,
                     rapid_r_deseas: 0.385, rapid_r_raw: 0.341 } },
]);

// A COMPLETED stage-2-only run: the codec resumed at its final step, so the
// metrics carry NO stage-1 loss records at all — only stage-2. This is what
// every E-012+ arm looks like in the archive, and it is the exact shape that
// vanished on completion (2026-08-12): renderLiveRun returned "" for
// completed runs before trying the stage-2 panel, so a row of successful 60k
// trainings rendered as "None of the last N runs produced curves".
const RUN_S2_DONE = jsonl([
  { config: { steps: 60000, params_M: 40.7, data: "f3_na025.npz", C: 39,
              resume: "run-62" } },
  { resumed: { from: "run-62.pt", parent_tag: "run-62", at_step: 60000 } },
  { stage2_config: { d_model: 576, layers: 8, K: 24, steps: 60000,
                     params_M: 32.038, batch: 256, train_windows: 28900000,
                     d_z: 64, seed: 2, tag: "" } },
  { stage2_step: 30000, stage2_zmse: 0.31, stage2_wall_s: 900,
    stage2_val_zmse: 0.70, stage2_amp: 0.93, stage2_grad_norm: 0.6 },
  { stage2_step: 60000, stage2_zmse: 0.197, stage2_wall_s: 1800,
    stage2_val_zmse: 0.598, stage2_amp: 0.95, stage2_grad_norm: 0.5 },
  { stage2_result: { d_model: 576, layers: 8, K: 24, steps: 60000,
                     params_M: 32.038, seed: 2, tag: "",
                     z_mse_model: 0.598, z_mse_persistence: 3.149,
                     chan_mse_model: 0.76, chan_mse_persistence: 3.1,
                     rapid_r_deseas: 0.462, rapid_r_raw: 0.41 } },
]);

const RUN_B = jsonl([
  { config: { steps: 60000, batch: 256, d_z: 64, params_M: 40.7, data: "f3_na025.npz", C: 39, T: 528, resume: "run-101" } },
  { resumed: { from: "run-101.pt", parent_tag: "run-101", at_step: 22500 } },
  { step: 22650, loss_rec: 0.690, loss_nei: 1.690 },
  { step: 35000, loss_rec: 0.660, loss_nei: 1.660 },
  { step: 35000, linear_r_deseas: 0.601, wall_s: 3200 },
]);

// A stage-2 job still in flight: 6,000 of 24,000 steps in 2,400 s, so 0.4 s
// per step and 7,200 s left. No stage2_result yet, which is what makes it an
// ETA case rather than a verdict case.
const RUN_S2_LIVE = jsonl([
  { config: { steps: 60000, params_M: 40.7, data: "f3_na025.npz", C: 39 } },
  { stage2_config: { d_model: 192, layers: 4, K: 24, steps: 24000, params_M: 1.822, d_z: 64, seed: 0, tag: "" } },
  { stage2_step: 3000, stage2_zmse: 0.900, stage2_wall_s: 1200 },
  { stage2_step: 6000, stage2_zmse: 0.800, stage2_wall_s: 2400 },
]);

// A CONTINUATION: 60,000 steps done by a previous run at peak 1e-3, now
// extended to 200,000 at 1e-4. The chart must show both schedules and the
// seam, because a resumed cosine that silently reads 0.0 is the failure this
// display exists to make visible.
const RUN_S2_CONT = jsonl([
  { config: { steps: 60000, params_M: 40.7, data: "f3_na025.npz", C: 39 } },
  { stage2_resumed: { from: "f3_s2_60k__temporal", at_step: 60000, to_step: 200000,
                      parent_run: "112", parent_steps: 60000, parent_lr: 0.001, lr: 0.0001 } },
  { stage2_config: { d_model: 192, layers: 4, K: 24, steps: 200000, params_M: 1.822, d_z: 64, lr: 0.0001 } },
  { stage2_step: 72000, stage2_zmse: 0.700, stage2_lr: 9.7e-5, stage2_wall_s: 5000 },
  { stage2_step: 84000, stage2_zmse: 0.680, stage2_lr: 9.4e-5, stage2_wall_s: 10000 },
]);

const RUNS = {
  workflow_runs: [
    {
      id: 7, run_number: 107, status: "in_progress", conclusion: null,
      created_at: iso(200), run_started_at: iso(195),
      html_url: "https://example.invalid/107",
      display_title: "E-008 continuation to 200k at 1/10th lr", name: "ml-train",
      head_sha: "7".repeat(40),
    },
    {
      id: 6, run_number: 106, status: "in_progress", conclusion: null,
      created_at: iso(45), run_started_at: iso(41),
      html_url: "https://example.invalid/106",
      display_title: "E-007 stage-2 budget sweep, 24,000 steps", name: "ml-train",
      head_sha: "6".repeat(40),
    },
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
      id: 8, run_number: 108, status: "completed", conclusion: "success",
      created_at: iso(20), html_url: "https://example.invalid/108",
      display_title: "E-017 capacity rung (stage-2 only)", name: "ml-train",
      head_sha: "f".repeat(40),
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
    if (/ml-metrics\/plan-105\.json/.test(url)) return fulfill(JSON.stringify({
      steps: 200000, lr: 1e-4, at_step: 60000,
      parent_run: 112, parent_steps: 60000, parent_lr: 1e-3,
    }));
    // #106's plan is a WARM RESTART, the shape actually dispatched for E-008:
    // `steps` is the EXTRA, so the axis must be parent + extra or a
    // 200,000-step experiment gets drawn on a 140,000-step axis.
    if (/ml-metrics\/plan-106\.json/.test(url)) return fulfill(JSON.stringify({
      steps: 140000, lr: 1e-4, warm: true,
      parent_run: 112, parent_steps: 60000, parent_lr: 1e-3,
    }));
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
    if (/ml-live-106\/metrics\.jsonl/.test(url)) return fulfill(RUN_S2_LIVE);
    if (/ml-live-107\/metrics\.jsonl/.test(url)) return fulfill(RUN_S2_CONT);
    if (/ml-metrics\/run-103\.jsonl/.test(url)) return fulfill(RUN_S2);
    if (/ml-metrics\/run-108\.jsonl/.test(url)) return fulfill(RUN_S2_DONE);
    if (/ml-metrics\/run-101\.jsonl/.test(url)) return fulfill(RUN_A);
    // #104 is mid-build and it is the case that actually bit: a live branch
    // that ALREADY has a metrics.jsonl (the `config` line is published within
    // seconds of the job starting) but no loss points yet. The plan preview
    // used to render only when the metrics file was entirely absent, i.e. for
    // the first few seconds of a run and never when anyone looked.
    if (/ml-metrics\/plan-104\.json/.test(url)) return fulfill(JSON.stringify({
      steps: 200000, lr: 1e-4, at_step: 60000,
      parent_run: 112, parent_steps: 60000, parent_lr: 1e-3,
    }));
    // ...plus embedding progress. The frozen-codec embedding is ~95 minutes of
    // a stage-2 run and reported nothing at all until 2026-08-10, so the page
    // showed one unchanging phase line for an hour and a half and a wedged job
    // looked exactly like a working one. Only the LAST record matters.
    if (/ml-live-104\/metrics\.jsonl/.test(url)) return fulfill(
      JSON.stringify({ config: { steps: 60000, batch: 512, d_z: 64 } }) + "\n" +
      JSON.stringify({ resumed: { from: "run-62.pt", at_step: 60000 } }) + "\n" +
      JSON.stringify({ embedding: { pct: 25.0, month: 129, months: 516, elapsed_s: 1400, eta_s: 4200, where: "ram" } }) + "\n" +
      JSON.stringify({ embedding: { pct: 73.1, month: 377, months: 516, elapsed_s: 4100, eta_s: 1500, where: "ram" } }) + "\n");
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

test("a COMPLETED stage-2-only run keeps its stage-2 card", async ({ page }) => {
  // #108 resumed the codec at its final step, so its archive has NO stage-1
  // loss records — only stage-2. The card must still render (curve, held-out
  // line, verdict). This is the regression of 2026-08-12: completed runs
  // returned "" before the stage-2 fallback, so every finished E-012+ arm
  // vanished from TRAINING CURVES the moment it succeeded.
  await page.goto("/status.html");
  const card = page.locator("#live .card")
    .filter({ has: page.locator("h3", { hasText: "run #108" }) });
  await expect(card).toHaveCount(1);
  await expect(card).toContainText("stage 2");
  await expect(card).toContainText("HELD-OUT");
  await expect(card.locator("svg.chart")).toHaveCount(1);
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
  // The four scale numbers (2026-08-11): parameters above, plus batch,
  // steps and data points, straight from the run's own stage2_config \u2014
  // an older run without batch/train_windows simply shows fewer fields.
  await expect(card).toContainText("batch 512");
  await expect(card).toContainText("6,000 steps");
  await expect(card).toContainText("26.1M windows");
  expect(await card.locator('svg.chart polyline[stroke="#a371f7"]').count()).toBe(1);

  // The z-space curve is charted against the persistence bar it must beat.
  await expect(card).toContainText("persistence 1.173");
  await expect(card).toContainText("latest 0.7810");
  // In-training monitoring (2026-08-11): held-out z-MSE with the amplitude
  // ratio (the smoothing diagnostic E-014 lacked), and the transport-probe
  // trend — older runs without these records simply show neither line.
  await expect(card).toContainText("HELD-OUT z-MSE");
  await expect(card).toContainText("0.8560");
  await expect(card).toContainText("amplitude ratio");
  await expect(card).toContainText("in-training RAPID");
  await expect(card).toContainText("0.21");

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

test("a run's doc links to its experiment definition", async ({ page }) => {
  // The doc strings open with an experiment ID, and the badge must point at
  // the EXPLICIT anchor in EXPERIMENTS.md — not GitHub's auto-generated
  // heading anchor, which is derived from the whole heading text and rots the
  // first time a heading changes from "READY" to a verdict.
  //
  // ...and at docs.html, not at the blob URL it used to use. CLAUDE.md SS0b:
  // a GitHub blob URL for markdown is "correct, unreadable on a phone", and
  // this is the badge a phone reader taps to find out what E-007 even was.
  await page.route(/\/actions\/workflows\/[^/]+\/runs/, (route) =>
    route.fulfill({
      status: 200, contentType: "application/json",
      body: JSON.stringify({ workflow_runs: [
        { id: 12, run_number: 112, status: "in_progress", conclusion: null,
          created_at: iso(20), run_started_at: iso(18),
          html_url: "https://example.invalid/112", name: "ml-train",
          display_title: "E-007 third point: stage-2 only, 60,000 steps",
          head_sha: "f".repeat(40) },
        { id: 11, run_number: 42, status: "completed", conclusion: "success",
          created_at: iso(400), html_url: "https://example.invalid/42",
          display_title: "an older run with no experiment id", name: "ml-train",
          head_sha: "9".repeat(40) },
      ] }),
    }));
  await page.goto("/status.html");
  const badge = page.locator("#runs a.exp");
  await expect(badge).toHaveCount(1);           // only the E-007 run gets one
  await expect(badge).toHaveText("E-007");
  await expect(badge).toHaveAttribute("href", /ml\/EXPERIMENTS\.md#e-007$/);
  await expect(badge).toHaveAttribute("href", /\/docs\.html\?f=ml\/EXPERIMENTS\.md#e-007$/);
  // and never the blob form again
  const href = await badge.getAttribute("href");
  expect(href).not.toMatch(/github\.com\/[^/]+\/[^/]+\/blob\//);
});

test("the docs badge and the Live app link follow the origin, in all three contexts",
  async ({ page }) => {
  // This assertion used to read `"https://blauewelt.github.io/earth/docs.html
  // ?f=ml/EXPERIMENTS.md#e-007"` — the literal origin, hardcoded in
  // status.html and echoed here on purpose as the tripwire that would notice
  // the day the site moved (docs/HOSTING.md SS8 listed it as exactly that).
  // The site is moving, so the tripwire moves with it rather than being
  // deleted: what must now hold is that NOTHING pins the page to one host.
  //
  // Three contexts, because status.html is opened from all three and the
  // right answer is different in each:
  //   1. the origin that served it        -> follow it
  //   2. some OTHER origin that served it -> follow that one instead
  //   3. a saved file:// copy             -> no origin to follow; fall back
  //      to the public GitHub Pages copy, which stays live indefinitely and
  //      is the rollback target
  // (2) is the one that matters for the cutover: earth.pages.dev and
  // blauewelt.org must send readers to their OWN docs.html, not to a
  // github.io copy that is only as fresh as its last Pages deploy. 127.0.0.1
  // is the same test server behind a different host string, which is all the
  // page can tell about "a different origin" anyway.
  await page.route(/\/actions\/workflows\/[^/]+\/runs/, (route) =>
    route.fulfill({
      status: 200, contentType: "application/json",
      body: JSON.stringify({ workflow_runs: [
        { id: 12, run_number: 112, status: "in_progress", conclusion: null,
          created_at: iso(20), run_started_at: iso(18),
          html_url: "https://example.invalid/112", name: "ml-train",
          display_title: "E-007 third point: stage-2 only, 60,000 steps",
          head_sha: "f".repeat(40) },
      ] }),
    }));

  const badge = page.locator("#runs a.exp");
  const liveApp = page.locator("#live-app");

  // 1 — the serving origin (localhost, the suite and any dev server).
  await page.goto("/status.html");
  await expect(badge).toHaveAttribute(
    "href", "http://localhost:8080/docs.html?f=ml/EXPERIMENTS.md#e-007");
  await expect(liveApp).toHaveAttribute("href", "http://localhost:8080/");

  // 2 — a different serving origin. Standing in for earth.pages.dev and
  //     blauewelt.org: same bytes, different host, links must follow.
  await page.goto("http://127.0.0.1:8080/status.html");
  await expect(badge).toHaveAttribute(
    "href", "http://127.0.0.1:8080/docs.html?f=ml/EXPERIMENTS.md#e-007");
  await expect(liveApp).toHaveAttribute("href", "http://127.0.0.1:8080/");

  // 3 — a saved copy on disk. A relative URL here would resolve to a path on
  //     the reader's own filesystem, so this case, and only this case, keeps
  //     the absolute GitHub Pages fallback.
  await page.goto(pathToFileURL(path.join(__dirname, "..", "status.html")).href);
  await expect(liveApp).toHaveAttribute("href", "https://blauewelt.github.io/earth/");
  await expect(badge).toHaveAttribute(
    "href", "https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-007");

  // The behaviour-preservation claim in docs/HOSTING.md SS8, checked rather
  // than asserted in prose: on GitHub Pages the origin-following branch
  // produces exactly the two strings that used to be hardcoded.
  const onPages = await page.evaluate(() => ({
    docs: new URL("docs.html", "https://blauewelt.github.io/earth/status.html").href,
    app: new URL("./", "https://blauewelt.github.io/earth/status.html").href,
  }));
  expect(onPages.docs).toBe("https://blauewelt.github.io/earth/docs.html");
  expect(onPages.app).toBe("https://blauewelt.github.io/earth/");
});

test("a long stage-2 job says when it will finish, not just how far along", async ({ page }) => {
  // 6,000 of 24,000 steps in 2,400 s -> 0.4 s/step -> 7,200 s left. The panel
  // must name a clock time: "25% done" on a seven-hour job is not something
  // you can plan around, and this is read on a phone.
  await page.goto("/status.html");
  const card = page.locator("#live .card")
    .filter({ has: page.locator("h3", { hasText: "run #106" }) });
  await expect(card).toContainText("6,000 of 24,000 steps");
  await expect(card).toContainText("~2.0 h left");
  await expect(card).toContainText("ends \u2248");
  // A FINISHED stage 2 states its verdict instead — no ETA once it is done.
  const done = page.locator("#live .card")
    .filter({ has: page.locator("h3", { hasText: "run #103" }) });
  await expect(done).not.toContainText("ends \u2248");
});

test("a continued stage-2 run charts BOTH learning-rate schedules and the seam", async ({ page }) => {

  // Chris: render the LR into the graph from the beginning, and the previous
  // run's schedule over steps 0-60k in the same picture, so the exact schedule
  // is readable rather than inferred.
  await page.goto("/status.html");
  const card = page.locator("#live .card")
    .filter({ has: page.locator("h3", { hasText: "run #107" }) });
  // "as LOGGED" because the curve is now the rate the run REPORTED rather
  // than one re-derived from an assumed cosine — the peak is still this run's
  // configured 1.0e-4, not the parent's 1.0e-3 and not the highest value
  // logged so far.
  await expect(card).toContainText("learning rate as LOGGED, peak 1.0e-4");
  await expect(card).toContainText("previous run's schedule (peak 1.0e-3, 60,000 steps)");
  // Two LR polylines — the parent's dashed, this run's solid — plus the seam.
  expect(await card.locator('svg.chart polyline[stroke="#f0883e"]').count()).toBe(2);
  expect(await card.locator('svg.chart polyline[stroke="#f0883e"][stroke-dasharray]').count()).toBe(1);
  expect(await card.locator('svg.chart line[stroke="#6e7681"][stroke-dasharray]').count()).toBe(1);
  // And the axis runs from 0, not from the seam: the whole trajectory.
  await expect(card).toContainText("200,000");
});

test("empty recent runs do not evict the runs that DO have curves", async ({ page }) => {
  // The exact shape of 2026-08-10: a burst of cancels and rescue-only jobs
  // sitting above the last run that actually plotted anything. Selecting "the
  // six most recent completed runs" showed six empties and hid the curve.
  const empties = [201, 202, 203, 204, 205, 206].map((n, i) => ({
    id: 900 + i, run_number: n, status: "completed",
    conclusion: n % 2 ? "cancelled" : "success",
    created_at: iso(60 + i), html_url: "https://example.invalid/" + n,
    display_title: "rescue-only / cancelled", name: "ml-train",
    head_sha: String(n).repeat(8).slice(0, 40),
  }));
  await page.route(/\/actions\/workflows\/[^/]+\/runs/, (route) =>
    route.fulfill({
      status: 200, contentType: "application/json",
      body: JSON.stringify({ workflow_runs: empties.concat([{
        id: 999, run_number: 103, status: "completed", conclusion: "success",
        created_at: iso(300), html_url: "https://example.invalid/103",
        display_title: "the run that actually has a stage-2 curve",
        name: "ml-train", head_sha: "c".repeat(40),
      }]) }),
    }));
  await page.goto("/status.html");
  const live = page.locator("#live");
  // #103's archive is stubbed in beforeEach and carries a full stage-2 run.
  await expect(live).toContainText("temporal transformer over the frozen embeddings");
  await expect(live).toContainText("6 more recent runs had nothing to plot");
});

test("a live run deep in the list still gets a card, and the table stays short",
  async ({ page }) => {
  // 2026-08-10: eight cancels and re-dispatches inside twenty minutes pushed
  // #121 — a 200,000-step CPU run with six hours left — past the twelfth
  // position, and the page fetched exactly twelve. It disappeared from the
  // dashboard while still burning a GPU, and a missing card reads exactly
  // like a finished job. My own watcher had the identical defect and went
  // quiet about the same run in the same minute, which is what made it
  // visible at all.
  //
  // The fetch window is 30 now while the table still lists 12, so this pins
  // both halves: the deep live run is CHARTED, and the table did not grow a
  // screenful on a phone.
  const filler = Array.from({ length: 18 }, (_, i) => ({
    id: 8000 + i, run_number: 200 + i, status: "completed", conclusion: "cancelled",
    created_at: iso(30 + i), html_url: "https://example.invalid/f" + i,
    display_title: "cancelled before it did anything", name: "ml-train",
    head_sha: String(i % 10).repeat(40).slice(0, 40),
  }));
  await page.route(/\/actions\/workflows\/[^/]+\/runs/, (route) =>
    route.fulfill({
      status: 200, contentType: "application/json",
      // The live run sits BELOW eighteen newer ones — outside the old window.
      body: JSON.stringify({ workflow_runs: filler.concat([{
        id: 106, run_number: 106, status: "in_progress", conclusion: null,
        created_at: iso(400), run_started_at: iso(400),
        html_url: "https://example.invalid/106",
        display_title: "the long CPU run everyone forgot about",
        name: "ml-train", head_sha: "e".repeat(40),
      }]) }),
    }));
  await page.goto("/status.html");
  // #106's live branch is stubbed in beforeEach with a stage-2 run.
  const card = page.locator("#live .card")
    .filter({ has: page.locator("h3", { hasText: "run #106" }) });
  await expect(card).toContainText("in progress");
  // …and the table did not become nineteen entries on a phone. Count the doc
  // rows: the table emits TWO <tr> per run (the row and its `.docrow`), so a
  // raw <tr> count is 2n+1 and asserting on it tests arithmetic rather than
  // behaviour — the first version of this line did exactly that and failed
  // against correct code.
  expect(await page.locator("#runs table tr.docrow").count()).toBe(12);
});

test("a run that has not started yet shows its PLANNED schedule", async ({ page }) => {
  // Chris: "plot the curve of the continuation run before it starts (incl LR)
  // ... then we know the exact schedule." A resumed cosine that reads 0.0 is
  // cheap to catch here and expensive to catch after sixteen hours.
  await page.goto("/status.html");
  const card = page.locator("#live .card")
    .filter({ has: page.locator("h3", { hasText: "run #105" }) });
  await expect(card).toContainText("planned schedule");
  await expect(card).toContainText("not yet run");
  await expect(card).toContainText("learning rate, peak 1.0e-4 from step 60,000");
  await expect(card).toContainText("previous run (peak 1.0e-3)");
  // Both segments and the seam, exactly as the live chart draws them.
  expect(await card.locator('svg polyline[stroke="#f0883e"]').count()).toBe(2);
  expect(await card.locator('svg line[stroke="#6e7681"][stroke-dasharray]').count()).toBe(1);
  // And it still says it is queued — the plan does not pretend to be progress.
  await expect(card).toContainText("queued");
});

test("a run that has STARTED but not trained yet still shows its plan", async ({ page }) => {
  // The regression this pins is the one Chris hit: "the dashboard has
  // nothing", with plan-117.json published and publicly readable the whole
  // time. planChart was reachable only from the `text === null` branch, and a
  // live branch stops being null the moment the job publishes its config
  // line — seconds in. So the preview existed, was correct, and was
  // unobservable. #104 here is in_progress, has a metrics file with no loss
  // points, and has a plan; all three of those are true of a resumed stage-2
  // job for the first half hour of its life.
  await page.goto("/status.html");
  const card = page.locator("#live .card")
    .filter({ has: page.locator("h3", { hasText: "run #104" }) });
  await expect(card).toContainText("planned schedule");
  await expect(card).toContainText("learning rate, peak 1.0e-4 from step 60,000");
  // The phase is still reported — the plan is added to the waiting card, not
  // substituted for it. A chart that replaced "building the tensor" would
  // trade one missing fact for another.
  await expect(card).toContainText("building the tensor");
  expect(await card.locator('svg polyline[stroke="#f0883e"]').count()).toBe(2);
});

test("a run rebuilding its embedding says how far along it is", async ({ page }) => {
  // Chris, watching #119: "didn't you say we're 70% done before?" The honest
  // answer was that the number came from watching the box's resident memory
  // climb, because embed_everything printed nothing for ninety-five minutes
  // and Actions will not serve the log of a running job. The phase line said
  // "probes and stage 2" the entire time, so a wedged job and a working one
  // rendered identically. Now the run reports its own progress.
  await page.goto("/status.html");
  const card = page.locator("#live .card")
    .filter({ has: page.locator("h3", { hasText: "run #104" }) });
  await expect(card).toContainText("rebuilding the frozen-codec embedding");
  // The LATEST record, not the first — this is a progress bar, not a series.
  await expect(card).toContainText("73.1%");
  await expect(card).not.toContainText("25%");
  await expect(card).toContainText("month 377 of 516");
  await expect(card).toContainText("~25 min left");
  // And it says the cache is not being written, which is why the NEXT run
  // will pay this cost again.
  await expect(card).toContainText("in RAM — not cached");
  const w = await card.locator(".bar i").first().evaluate((el) => el.style.width);
  expect(w).toBe("73.1%");
});

test("a warm restart is charted as a restart, not as a continuation", async ({ page }) => {
  // #120 takes the 60k head's WEIGHTS and trains 140,000 more on a FRESH
  // cosine; #121 trains 200,000 from scratch. Comparing their curves is the
  // point of running both, so the page has to draw the warm restart honestly:
  // same axis of total compute, but its own peak at the seam rather than
  // entering the parent's cosine partway down. Drawing them identically would
  // hide the one distinction the experiment turns on.
  await page.route(/ml-live-107\/metrics\.jsonl/, (route) => route.fulfill({
    status: 200, contentType: "text/plain",
    body: [
      JSON.stringify({ config: { steps: 60000, params_M: 40.7, C: 39 } }),
      JSON.stringify({ stage2_warm_restart: {
        from: "f3_s2_60k__temporal.pt", parent_steps: 60000, parent_lr: 0.001,
        extra_steps: 140000, lr: 0.0001,
        reset: ["optimiser moments", "schedule position", "rng stream"],
        kind: "warm restart (cosine restart), NOT a continuation" } }),
      JSON.stringify({ stage2_config: { d_model: 192, layers: 4, K: 24,
                                        steps: 140000, params_M: 1.822, lr: 0.0001 } }),
      JSON.stringify({ stage2_step: 3000, stage2_zmse: 0.70, stage2_lr: 9.9e-5, stage2_wall_s: 1300 }),
      JSON.stringify({ stage2_step: 6000, stage2_zmse: 0.69, stage2_lr: 9.8e-5, stage2_wall_s: 2600 }),
    ].join("\n"),
  }));
  await page.goto("/status.html");
  const card = page.locator("#live .card")
    .filter({ has: page.locator("h3", { hasText: "run #107" }) });
  // TEXT ASSERTIONS FIRST, because they auto-wait and `locator.count()` does
  // not: counting before the card has rendered returns 0 and reads exactly
  // like a missing chart. This test failed that way once before the order was
  // fixed, which is the kind of flake that gets a real regression dismissed.
  //
  // The legend must name this run's OWN peak and the parent's, and must say
  // what this is — captioning a warm restart as a continuation would be a
  // false claim in the one place a reader looks to check the claim.
  await expect(card).toContainText("warm restart");
  await expect(card).toContainText("1.0e-4");
  await expect(card).toContainText("1.0e-3");
  // Two schedule segments: the parent dashed, this run solid.
  expect(await card.locator('svg polyline[stroke="#f0883e"]').count())
    .toBeGreaterThanOrEqual(2);
});

test("a warm restart's PLAN is drawn on the axis of total compute", async ({ page }) => {
  // The plan preview exists to be checked before a run spends anything, so
  // getting its axis wrong defeats the point. For a warm restart `steps` is
  // the EXTRA (140,000), and the run occupies 60,000..200,000 of total
  // compute. Drawing it on a 0..140,000 axis would misplace every point and
  // make it incomparable with the from-scratch sibling it exists to be
  // compared against.
  await page.route(/ml-live-106\/metrics\.jsonl/, (route) =>
    route.fulfill({ status: 200, contentType: "text/plain",
                    body: JSON.stringify({ config: { steps: 60000 } }) + "\n" }));
  await page.goto("/status.html");
  const card = page.locator("#live .card")
    .filter({ has: page.locator("h3", { hasText: "run #106" }) });
  await expect(card).toContainText("planned schedule");
  await expect(card).toContainText("warm restart");
  // The axis label is TOTAL compute, not the extra steps.
  await expect(card).toContainText("200,000");
  await expect(card).not.toContainText("learning rate, peak 1.0e-4 from step");
  // Parent dashed + this run solid, and a seam between them.
  expect(await card.locator('svg polyline[stroke="#f0883e"]').count()).toBe(2);
  expect(await card.locator('svg line[stroke="#6e7681"][stroke-dasharray]').count()).toBe(1);
});

test("a plan carrying POINTS is drawn from them, not re-derived", async ({ page }) => {
  // Chris: "the queueing graph needs to be computed from the actual
  // implementation." Re-deriving the curve here is a second implementation of
  // the schedule — it would have gone on drawing a cosine for a wsd run
  // without a word, certifying a schedule the run does not use. The points
  // come from ml/plan_schedule.py sampling the trainer's own make_sched.
  const pts = [];
  for (let i = 0; i <= 200; i++) {
    const s = Math.round(i * 199999 / 200);
    const lr = s <= 2000 ? 1e-3 * (s + 1) / 2000
             : s <= 180000 ? 1e-3
             : 1e-3 * Math.max(0, (200000 - s) / 20000);
    pts.push([s, lr]);
  }
  await page.route(/ml-metrics\/plan-106\.json/, (r) => r.fulfill({
    status: 200, contentType: "text/plain",
    body: JSON.stringify({ steps: 200000, lr: 1e-3, schedule: "wsd",
                           cooldown_frac: 0.1, points: pts }) }));
  await page.route(/ml-live-106\/metrics\.jsonl/, (r) => r.fulfill({
    status: 200, contentType: "text/plain",
    body: JSON.stringify({ config: { steps: 60000 } }) + "\n" }));
  await page.goto("/status.html");
  const card = page.locator("#live .card")
    .filter({ has: page.locator("h3", { hasText: "run #106" }) });
  await expect(card).toContainText("computed from the trainer's own scheduler");
  await expect(card).toContainText("wsd, peak 1.0e-3");
  await expect(card).toContainText("200,000");
  // One segment (no parent), and it must be the points — a re-derived cosine
  // would not hold a flat top, so sample the drawn polyline for the plateau.
  const line = card.locator('svg polyline[stroke="#f0883e"]');
  await expect(line).toHaveCount(1);
  const d = await line.getAttribute("points");
  const ys = d.split(" ").map((q) => parseFloat(q.split(",")[1]));
  const mid = ys.slice(30, 150);
  expect(Math.max(...mid) - Math.min(...mid)).toBeLessThan(0.5);  // flat stable phase
  expect(ys[ys.length - 1]).toBeGreaterThan(ys[80] + 20);          // then it cools down
});

test("the head k-fold leads, and the 36-month split is labelled as such", async ({ page }) => {
  // Added 2026-08-10, the day probe_kfold turned out to score the CODEC:
  // #116 (60k head) and #125 (200k head) returned bit-identical RAPID
  // 0.631, because that probe never sees the temporal transformer. Until
  // rapid_probe_kfold existed, every stage-2 comparison rested on a single
  // split of 36 held-out months with an SE of order 0.15 — and this panel
  // printed that number bare, with nothing saying how thin it was.
  //
  // A run carrying BOTH must lead with the k-fold and mark the split, or the
  // noisy figure is what gets read off a phone and quoted.
  await page.route(/ml-live-106\/metrics\.jsonl/, (r) => r.fulfill({
    status: 200, contentType: "text/plain",
    body: [
      JSON.stringify({ config: { steps: 60000, params_M: 40.7, C: 39 } }),
      JSON.stringify({ stage2_config: { d_model: 192, layers: 4, K: 24, steps: 6000 } }),
      JSON.stringify({ stage2_step: 1500, stage2_zmse: 0.910, stage2_wall_s: 40 }),
      JSON.stringify({ stage2_step: 3000, stage2_zmse: 0.845, stage2_wall_s: 80 }),
      JSON.stringify({ stage2_step: 6000, stage2_zmse: 0.781, stage2_wall_s: 160 }),
      JSON.stringify({ stage2_result: {
        d_model: 192, layers: 4, K: 24, steps: 6000, params_M: 1.8, seed: 0,
        z_mse_model: 0.731, z_mse_persistence: 1.173,
        rapid_r_deseas: 0.317, rapid_r_raw: 0.392,
        rapid_r_kfold: 0.462, rapid_r_kfold_ci: [0.331, 0.577] } }),
    ].join("\n") + "\n" }));
  await page.goto("/status.html");
  const card = page.locator("#live .card")
    .filter({ has: page.locator("h3", { hasText: "run #106" }) });
  await expect(card).toContainText("RAPID r +0.462 [0.33, 0.58] (k-fold)");
  await expect(card).toContainText("RAPID r +0.317 (36-mo split)");
  // The k-fold must come FIRST in the sentence, not merely be present.
  // Pick the .probe that IS the stage-2 verdict — the card carries more than
  // one, and `.first()` silently grabbed the stage-1 panel, where both
  // indexOf calls returned -1 and the comparison was meaningless in a way
  // that still reads like a real assertion.
  const txt = await card.locator(".probe")
    .filter({ hasText: "stage-2 result" }).innerText();
  expect(txt.indexOf("(k-fold)")).toBeGreaterThan(-1);
  expect(txt.indexOf("(k-fold)")).toBeLessThan(txt.indexOf("(36-mo split)"));
  // And the panel must say why the OTHER k-fold on this page cannot move.
  await expect(card).toContainText("scores the");
  await expect(card).toContainText("CODEC");
});

test("a run predating the head k-fold says its RAPID number is a single split",
  async ({ page }) => {
  await page.route(/ml-live-106\/metrics\.jsonl/, (r) => r.fulfill({
    status: 200, contentType: "text/plain",
    body: [
      JSON.stringify({ config: { steps: 60000, params_M: 40.7, C: 39 } }),
      JSON.stringify({ stage2_config: { d_model: 192, layers: 4, K: 24, steps: 6000 } }),
      JSON.stringify({ stage2_step: 1500, stage2_zmse: 0.910, stage2_wall_s: 40 }),
      JSON.stringify({ stage2_step: 3000, stage2_zmse: 0.845, stage2_wall_s: 80 }),
      JSON.stringify({ stage2_step: 6000, stage2_zmse: 0.781, stage2_wall_s: 160 }),
      JSON.stringify({ stage2_result: { steps: 6000, z_mse_model: 0.731,
                                        z_mse_persistence: 1.173,
                                        rapid_r_deseas: 0.317 } }),
    ].join("\n") + "\n" }));
  await page.goto("/status.html");
  const card = page.locator("#live .card")
    .filter({ has: page.locator("h3", { hasText: "run #106" }) });
  // No "(36-mo split)" suffix when there is nothing to distinguish it from —
  // the label would be noise. The prose carries the caveat instead.
  await expect(card).toContainText("RAPID r +0.317");
  await expect(card).toContainText("predates the head k-fold");
  await expect(card).not.toContainText("(k-fold)");
});

test("records from a PREVIOUS run are discarded, not charted", async ({ page }) => {
  // On 2026-08-10 the rescue step copied the previous job's metrics.jsonl
  // without removing it, so #123's temporal.py appended to #120's file and
  // its live branch opened with a stage-2 step belonging to another run. It
  // fooled this page, the monitor and a verification script at once — caught
  // only because the stale record's stage2_wall_s exceeded #123's whole
  // lifetime. A `config` line is a run's first line; anything before it is
  // somebody else's.
  await page.route(/ml-live-106\/metrics\.jsonl/, (r) => r.fulfill({
    status: 200, contentType: "text/plain",
    body: [
      // ---- the previous run's leftovers ----
      JSON.stringify({ stage2_step: 7000, stage2_zmse: 0.82014, stage2_lr: 9.9384e-5, stage2_wall_s: 2657.7 }),
      JSON.stringify({ stage2_config: { d_model: 192, layers: 4, K: 24, steps: 140000 } }),
      JSON.stringify({ stage2_warm_restart: { from: "old.pt", parent_steps: 60000, parent_lr: 1e-3, extra_steps: 140000, lr: 1e-4 } }),
      // ---- THIS run starts here ----
      JSON.stringify({ config: { steps: 60000, batch: 512, d_z: 64 } }),
      JSON.stringify({ embedding: { pct: 42.0, month: 217, months: 516, elapsed_s: 2400, eta_s: 3300, where: "disk" } }),
    ].join("\n") }));
  await page.goto("/status.html");
  const card = page.locator("#live .card")
    .filter({ has: page.locator("h3", { hasText: "run #106" }) });
  // It must show THIS run's state — still embedding …
  await expect(card).toContainText("rebuilding the frozen-codec embedding");
  await expect(card).toContainText("42%");
  // … and must NOT have charted the other run's stage 2. Assert on things
  // only the stale RECORDS could produce: the loss value, and the stage-2
  // panel itself. ("warm restart" would be the wrong assertion — this
  // fixture's PLAN is legitimately a warm restart, and the point is to catch
  // contamination from metrics, not to forbid a word.)
  await expect(card).not.toContainText("0.8201");
  await expect(card).not.toContainText("z-space MSE");
  await expect(card).not.toContainText("temporal transformer over the frozen");
});

test("the live LR overlay plots the LOGGED rate, not a re-derived cosine", async ({ page }) => {
  // #125 runs an exponential decay. cosLr would have drawn a cosine over it,
  // on a chart whose legend named the schedule correctly — the same "second
  // implementation" defect the plan preview had. temporal.py logs stage2_lr
  // every point, so there is nothing to derive.
  const body = [
    JSON.stringify({ config: { steps: 60000, params_M: 40.7 } }),
    JSON.stringify({ stage2_config: { d_model: 192, layers: 4, K: 24, steps: 200000, params_M: 1.822 } }),
    // An exponential tail: a cosine over 200k would still be near its peak here.
    JSON.stringify({ stage2_step: 2000, stage2_zmse: 1.98, stage2_lr: 1.0e-3, stage2_wall_s: 51 }),
    JSON.stringify({ stage2_step: 40000, stage2_zmse: 0.71, stage2_lr: 5.0e-4, stage2_wall_s: 1030 }),
    JSON.stringify({ stage2_step: 80000, stage2_zmse: 0.55, stage2_lr: 2.5e-4, stage2_wall_s: 2060 }),
  ].join("\n");
  await page.route(/ml-live-106\/metrics\.jsonl/, (r) =>
    r.fulfill({ status: 200, contentType: "text/plain", body }));
  await page.route(/ml-metrics\/plan-106\.json/, (r) => r.fulfill({
    status: 200, contentType: "text/plain",
    body: JSON.stringify({ steps: 200000, lr: 1e-3, schedule: "expdecay",
                           points: [[0, 1e-3], [100000, 1.7e-4], [199999, 3.2e-5]] }) }));
  await page.goto("/status.html");
  const card = page.locator("#live .card")
    .filter({ has: page.locator("h3", { hasText: "run #106" }) });
  await expect(card).toContainText("expdecay");
  await expect(card).toContainText("as LOGGED");
  // The observed segment must reproduce the halving, which a cosine cannot:
  // at step 80,000 of 200,000 a cosine at peak 1e-3 is 6.5e-4, not 2.5e-4.
  const lines = card.locator('svg polyline[stroke="#f0883e"]');
  expect(await lines.count()).toBeGreaterThanOrEqual(1);
  const pts = await lines.first().getAttribute("points");
  const ys = pts.split(" ").map((q) => parseFloat(q.split(",")[1]));
  // y is inverted: 1e-3 at the top, 2.5e-4 near the bottom of the box.
  expect(ys[ys.length - 1]).toBeGreaterThan(ys[0] + 10);
});

test("a rolled evaluation shows a progress bar and an ETA, not a blank card", async ({ page }) => {
  // Chris, 2026-08-14: "Do you have any sense of progress or an expected end
  // time on these evals? ... a progress bar would be helpful." He could not be
  // told and neither could I: a rolled eval trains nothing, so it published no
  // curve, and Actions will not serve a running job's log. The failure mode
  // this pins is the one that existed for #228/#229 — hours of GPU showing as
  // an empty panel — plus the subtler one it could have become: an sroll
  // record has no `step`, so falling through to the probe array would chart it
  // at x = undefined instead of being caught here.
  const body = [
    JSON.stringify({ config: { steps: 60000, params_M: 40.7 } }),
    JSON.stringify({ sroll: { head: "s1_s0", head_i: 1, heads: 4, phase: "skill",
                              done: 100, total: 714, pct: 14.0,
                              elapsed_s: 300, eta_head_s: 1800, eta_all_s: 9000 } }),
    JSON.stringify({ sroll: { head: "s9r222_s0", head_i: 2, heads: 4, phase: "long",
                              done: 357, total: 714, pct: 50.0,
                              elapsed_s: 4200, eta_head_s: 1050, eta_all_s: 5250 } }),
  ].join("\n");
  await page.route(/ml-live-106\/metrics\.jsonl/, (r) =>
    r.fulfill({ status: 200, contentType: "text/plain", body }));
  await page.goto("/status.html");
  const card = page.locator("#live .card")
    .filter({ has: page.locator("h3", { hasText: "run #106" }) });
  await expect(card).toContainText("Rolled evaluation");
  // the LATEST record wins — a progress bar is not a series
  await expect(card).toContainText("head 2 of 4");
  await expect(card).toContainText("s9r222_s0");
  await expect(card).toContainText("357 of 714");
  // and it says when to come back, which is the whole point of the request
  await expect(card).toContainText(/1 h|87 min|min/);
  // overall progress counts finished heads, not just this one: 1 done + half
  // of the second, over four = 37.5%
  const bar = card.locator('div[style*="background:#58a6ff"]').first();
  const w = await bar.getAttribute("style");
  expect(w).toMatch(/width:37\.5%/);
});

/* ---- per-run deep links --------------------------------------------------
 *
 * Session reports quote runs as bare numbers ("#413"). Until now the only
 * link that could be pasted went to the Actions page — a log, no curve — and
 * there was no way at all to link to a run ON this page. These four tests
 * cover the address (the id), the arrival (scroll + flash), the honest
 * failure (a hash naming a run that is not here), and the grab-handle.
 */

test("every run card carries its own anchor id, and only one owns it", async ({ page }) => {
  await page.goto("/status.html");
  await expect(page.locator("#live .card").first()).toBeVisible();
  const ids = await page.locator("#live .card").evaluateAll((els) =>
    els.map((e) => e.id));
  expect(ids.length).toBeGreaterThan(2);
  for (const id of ids) expect(id).toMatch(/^run-\d+$/);
  // Duplicate ids are the failure mode that matters: getElementById returns
  // whichever came first, so the second card becomes unreachable in silence.
  expect(new Set(ids).size).toBe(ids.length);
  // ...and the table must NOT also claim them.
  const tableIds = await page.locator("#runs [id]").evaluateAll((els) =>
    els.map((e) => e.id).filter((i) => /^run-/.test(i)));
  expect(tableIds).toEqual([]);
});

test("#run-NNN scrolls to that run's card and flashes it", async ({ page }) => {
  await page.goto("/status.html");
  await expect(page.locator("#run-101")).toBeAttached();
  // The flash clears itself after 1.4 s, so the hash change and the read of
  // the class happen in ONE evaluate — a round trip can outlast it on the
  // slow sandbox (CLAUDE.md SS4).
  const r = await page.evaluate(() => {
    const before = window.scrollY;
    location.hash = "#run-101";
    window.dispatchEvent(new HashChangeEvent("hashchange"));
    const el = document.getElementById("run-101");
    return { before, flashed: el.classList.contains("flash"),
             note: getComputedStyle(document.getElementById("runnote")).display };
  });
  expect(r.flashed).toBe(true);
  expect(r.note).toBe("none");          // a card that IS here explains nothing
  // and the page actually moved to it (smooth scrolling, so poll)
  await expect.poll(async () =>
    Math.abs(await page.evaluate(() =>
      document.getElementById("run-101").getBoundingClientRect().top)),
    { timeout: 5000 }).toBeLessThan(120);
});

test("a hash naming a run that is NOT on the page says so, and says where to look",
  async ({ page }) => {
    // The page fetches a window of runs. #386 fell out of it long ago, and a
    // hash that resolves to nothing is indistinguishable from a broken link —
    // the same reason SS4b makes a dateless layer announce itself.
    await page.goto("/status.html#run-386");
    const note = page.locator("#runnote");
    await expect(note).toBeVisible();
    await expect(note).toContainText("#386");
    await expect(note).toContainText("not on this page");
    // and it points at documents that DO have it, in the phone-readable form
    const hrefs = await note.locator("a").evaluateAll((as) =>
      as.map((a) => a.getAttribute("href")));
    expect(hrefs.join(" ")).toContain("docs.html?f=ml/RUNS.md#run-386");
    expect(hrefs.join(" ")).toContain("docs.html?f=ml/EXPERIMENTS.md");
    for (const h of hrefs) expect(h).not.toMatch(/\/blob\//);
    // no card was invented for it
    await expect(page.locator("#run-386")).toHaveCount(0);
  });

test("a run inside the window but with nothing to chart gets its own explanation",
  async ({ page }) => {
    // Two different absences, two different stories: "older than the window"
    // and "here, but cancelled before it plotted anything". #333 below is
    // fetched, listed in the table, and has no metrics file at all.
    await page.route(/\/actions\/workflows\/[^/]+\/runs/, (route) =>
      route.fulfill({
        status: 200, contentType: "application/json",
        body: JSON.stringify({ workflow_runs: [
          { id: 333, run_number: 333, status: "completed", conclusion: "cancelled",
            created_at: iso(60), html_url: "https://example.invalid/333",
            display_title: "cancelled before it trained anything", name: "ml-train",
            head_sha: "3".repeat(40) },
        ] }),
      }));
    await page.goto("/status.html#run-333");
    const note = page.locator("#runnote");
    await expect(note).toBeVisible();
    await expect(note).toContainText("#333");
    await expect(note).toContainText("no card");
    await expect(page.locator("#run-333")).toHaveCount(0);
  });

test("the permalink handle puts the run's address in the URL", async ({ page }) => {
  await page.goto("/status.html");
  await expect(page.locator("#run-101")).toBeAttached();
  await page.locator("#run-101 a.plink").click();
  await expect.poll(() => page.evaluate(() => location.hash)).toBe("#run-101");
  // and it is a real link, so a long-press "copy link address" works too
  await expect(page.locator("#run-101 a.plink")).toHaveAttribute("href", "#run-101");
});

/* ---- the structured config strip -----------------------------------------
 *
 * Standing rule, Chris 2026-08-19: *"make it a standing rule to have all
 * experiments descriptions be a) Structured, with fields: Num params, stage:
 * encoder, data, ... (you could even automatically render an experiment
 * config). b) Absolute, not relative."*
 *
 * These four tests pin the machine half. The order is fixed so a reader can
 * find a field by position; the SOURCES are the run's own records, so the
 * strip cannot drift from what ran; and a field with no record is OMITTED
 * rather than defaulted, because a zero a reader believes is worse than a gap
 * a reader can see.
 */

// probes-109.json, cut to the two keys the page reads. `inputs` is the
// verbatim dispatch block — strings, all of them, including "temporal_steps":
// "0", which is TRUE for an eval and which JS reads as truthy unless it is
// converted to a number first. That is the bug this fixture exists to catch.
const PROBES_109 = {
  run_number: 109,
  files: {
    "rollout_spatial.json": { heads: [] },
    "provenance.json": {
      run_number: 109,
      sha: "b".repeat(40),
      inputs: {
        doc: "E-035 SEED-PAIR EVAL: roll the freshly published seed-0 head.",
        steps: "60000", batch: "512", d_z: "64", anomaly: "true",
        temporal_steps: "0", temporal_d_model: "576", temporal_layers: "8",
        eval_every: "0", resume: "!run-62,run-63", head_probe: "false",
        window: "sroll:e017_u1_s0,head-weights-e035a-xl233-s0",
        tensor: "family3_na025", patch: "3", runner: "gpu-box-45731106",
        codec_d_model: "576", codec_layers: "10", codec_heads: "8",
        codec_d_dec: "768",
      },
    },
  },
};

// A finished ROLLED EVALUATION. It trains nothing, so its archived metrics
// carry no `config` line at all — no params, no batch, no tensor. Everything
// on its strip has to come from the dispatch block in probes-109.json, and
// `window` is the only field anywhere that separates an sroll job from a
// headpub one (both are "resumed at their final step with nothing to train").
const RUN_SROLL_DONE = jsonl([
  { sroll: { head: "s1_s0", head_i: 2, heads: 2, phase: "skill", done: 714,
             total: 714, pct: 100.0, elapsed_s: 2100, eta_head_s: 0, eta_all_s: 0 } },
]);

test("a finished run's config strip is rendered from its own records, in the fixed order",
  async ({ page }) => {
    await page.goto("/status.html");
    const card = page.locator("#live .card")
      .filter({ has: page.locator("h3", { hasText: "run #108" }) });
    const strip = card.locator(".cfg");
    await expect(strip).toHaveCount(1);
    // The order is the contract. It never varies, so the eye can find a field
    // by position on a card it has not read yet.
    const keys = await strip.locator(".k").evaluateAll((els) =>
      els.map((e) => e.textContent));
    expect(keys).toEqual(["params", "stage", "data", "arch", "steps×batch", "resume"]);
    // ...and every value is a number the run itself wrote. 40.7 is the codec's
    // params_M from the `config` line, 32.038 the head's from `stage2_config`;
    // neither is re-derived from d_model here, because re-deriving it would
    // restate the architecture in a second language and be wrong the first
    // time a channel is added.
    await expect(strip).toContainText("40.7M codec + 32.038M head");
    await expect(strip).toContainText("stage-2");
    await expect(strip).toContainText("f3_na025");
    await expect(strip).toContainText("head 576×8");
    await expect(strip).toContainText("resume");
    await expect(strip).toContainText("run-62");
    // the .npz is noise; C 39 is a fact and travels
    await expect(strip).toContainText("C 39");
  });

test("an eval that trained nothing takes its stage from probes-<n>.json, and omits the rest",
  async ({ page }) => {
    // The ambiguity that forces the provenance fetch: an sroll job and a
    // headpub job both resume at their final step and both train nothing.
    // Only `window` tells them apart, and `window` exists nowhere but the
    // dispatch block. Without it the strip would have to print a stage it
    // cannot know — which is the exact failure the "absolute" half of the
    // rule is against.
    const probeUrls = [];
    await page.route(/\/actions\/workflows\/[^/]+\/runs/, (route) =>
      route.fulfill({
        status: 200, contentType: "application/json",
        body: JSON.stringify({ workflow_runs: [
          { id: 9, run_number: 109, status: "completed", conclusion: "success",
            created_at: iso(25), html_url: "https://example.invalid/109",
            display_title: "E-035 SEED-PAIR EVAL: roll the seed-0 head",
            name: "ml-train", head_sha: "9".repeat(40) },
        ] }),
      }));
    await page.route(/ml-metrics\/run-109\.jsonl/, (r) =>
      r.fulfill({ status: 200, contentType: "text/plain", body: RUN_SROLL_DONE }));
    await page.route(/ml-metrics\/probes-109\.json/, (r) => {
      probeUrls.push(r.request().url());
      return r.fulfill({ status: 200, contentType: "application/json",
                         body: JSON.stringify(PROBES_109) });
    });
    await page.goto("/status.html");
    const card = page.locator("#live .card")
      .filter({ has: page.locator("h3", { hasText: "run #109" }) });
    const strip = card.locator(".cfg");
    await expect(strip).toHaveCount(1);
    await expect(strip).toContainText("sroll");
    // The dispatch block's fields, absolutely: the tensor by name, the codec
    // shape, the steps it was given and the resume mark VERBATIM. The "!" is
    // not decoration — train.py reads it as require_resume, so "!run-62" is a
    // job that fails rather than quietly training from scratch, and the
    // config line does not keep it.
    await expect(strip).toContainText("family3_na025");
    await expect(strip).toContainText("codec 576×10, 8 heads, d_dec 768");
    await expect(strip).toContainText("60,000 × 512");
    await expect(strip).toContainText("!run-62,run-63");
    const keys = await strip.locator(".k").evaluateAll((els) =>
      els.map((e) => e.textContent));
    // No params: this run has no params_M record anywhere, and the field is
    // simply ABSENT rather than 0 or a number recomputed from d_model.
    expect(keys).not.toContain("params");
    // ...and the fields it DOES have are still in the fixed order.
    expect(keys).toEqual(["stage", "data", "arch", "steps×batch", "resume"]);
    // temporal_steps is the string "0" — no head shape may be quoted for a
    // network this run never built, and no ZERO may stand in for one either.
    // The first draft of this strip rendered "head 0×0" and "head 0" here,
    // because `guard && obj.field` collapses to the boolean false when the
    // guard fails and Number(false) is 0. A reader would have read that as a
    // measurement. Assert the shape is absent, not merely different.
    const stripText = await strip.innerText();
    expect(stripText).not.toMatch(/head \d/);   // "8 heads" is fine; "head 0" is not
    expect(stripText).not.toMatch(/0×0/);
    expect(probeUrls.length).toBeGreaterThan(0);
  });

test("an in-flight run's strip shows only its live config line, and costs no extra fetch",
  async ({ page }) => {
    // probes-<n>.json is written by the workflow's final archive step, the same
    // one that deletes the live branch, so a running job HAS none. Asking for
    // it every two minutes would be a request that can only 404 — and the live
    // `config` line already carries enough to say something true.
    const probeUrls = [];
    await page.route(/ml-metrics\/probes-\d+\.json/, (r) => {
      probeUrls.push(r.request().url());
      return r.fulfill({ status: 404, body: "" });
    });
    await page.goto("/status.html");
    const card = page.locator("#live .card")
      .filter({ has: page.locator("h3", { hasText: "run #106" }) });
    const strip = card.locator(".cfg");
    await expect(strip).toHaveCount(1);
    const keys = await strip.locator(".k").evaluateAll((els) =>
      els.map((e) => e.textContent));
    // RUN_S2_LIVE's config line records steps, params_M, data and C — and no
    // batch and no resume. Those two fields are therefore absent, not zero
    // and not "none": nothing recorded them.
    expect(keys).toEqual(["params", "stage", "data", "arch", "steps×batch"]);
    expect(keys).not.toContain("resume");
    await expect(strip).toContainText("40.7M codec + 1.822M head");
    await expect(strip).toContainText("head 192×4");
    await expect(strip).toContainText("60,000");
    await expect(strip).not.toContainText("× 512");
    // no run in this fixture is finished-and-unresolvable, so nothing asked
    expect(probeUrls).toEqual([]);
  });

test("the header row is the card's anchor: the run number links to the card itself",
  async ({ page }) => {
    await page.goto("/status.html");
    const card = page.locator("#run-101");
    await expect(card).toBeAttached();
    const head = card.locator("h3");
    // one affordance, not two: the run NUMBER is the permalink handle. There
    // used to be a separate floated "#" beside a run number that linked to
    // Actions, which put the more prominent of two links on the surface
    // ml/CLAUDE.md §0c calls the wrong default.
    await expect(head.locator("a.plink")).toHaveCount(1);
    await expect(head.locator("a.plink")).toHaveAttribute("href", "#run-101");
    await expect(head.locator("a.plink")).toHaveText("run #101");
    // #NNN · E-tag · what the run DOES — the §0c form, rendered
    const card102 = page.locator("#run-102");
    await expect(card102.locator("h3")).toContainText("run #102");
    await expect(card102.locator("h3 .htitle")).toContainText("f3_anchor41M continued");
    // the Actions log is still reachable, just no longer the headline
    await expect(head.locator('a[href="https://example.invalid/101"]')).toHaveCount(1);
    // and the card's own id is what the header points at
    const [href, id] = await page.evaluate(() => {
      const c = document.getElementById("run-101");
      return [c.querySelector("h3 a.plink").getAttribute("href"), c.id];
    });
    expect(href).toBe("#" + id);
  });

test("nothing on a run card is a blob URL", async ({ page }) => {
    // CLAUDE.md §0b and ml/CLAUDE.md §0c: markdown goes to docs.html, never to
    // a GitHub blob — and no object URLs are minted for anything here either,
    // since a blob: href is unshareable by construction and this whole feature
    // is about addresses a reader can paste.
    await page.goto("/status.html");
    await expect(page.locator("#live .card").first()).toBeVisible();
    const hrefs = await page.locator("#live a").evaluateAll((as) =>
      as.map((a) => a.getAttribute("href") || ""));
    expect(hrefs.length).toBeGreaterThan(0);
    for (const h of hrefs) {
      expect(h).not.toMatch(/^blob:/);
      expect(h).not.toMatch(/github\.com\/[^/]+\/[^/]+\/blob\//);
    }
    // no src attribute anywhere in the live section mints one either
    const srcs = await page.locator("#live [src]").evaluateAll((els) =>
      els.map((e) => e.getAttribute("src") || ""));
    for (const s of srcs) expect(s).not.toMatch(/^blob:/);
  });

// ---- the API budget ---------------------------------------------------------
// 2026-08-21: Chris opened the page to "GitHub rate limit hit — retrying at :09"
// over four sections all reading "loading…". Unauthenticated api.github.com is
// 60 requests/hour/IP, the page spent exactly 60 (2 calls x 30 two-minute
// cycles), and a single 403 rejected the ONE Promise.all that also held the
// fleet and run-doc fetches — which come from raw.githubusercontent.com and are
// not rate limited at all. The page did not fail to render; it refused to.
//
// Headers below are stubbed exactly as GitHub sends them: x-ratelimit-reset is
// a UTC epoch in SECONDS, and all three are listed in GitHub's real
// Access-Control-Expose-Headers (checked against the live API the same day),
// so the expose header is stubbed too or the page could not read them.
const RATE_CORS = {
  "access-control-allow-origin": "*",
  "access-control-expose-headers":
    "X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset",
  "content-type": "application/json; charset=utf-8",
};
const hhmm = (ms) => {
  const d = new Date(ms);
  const p = (n) => (n < 10 ? "0" : "") + n;
  return p(d.getHours()) + ":" + p(d.getMinutes());
};

// Every api.github.com call answers 403 with an exhausted budget.
async function routeExhausted(page, resetMs, counter) {
  await page.route(/https:\/\/api\.github\.com\/.*/, async (route) => {
    if (counter) counter.push(route.request().url());
    await route.fulfill({
      status: 403,
      headers: {
        ...RATE_CORS,
        "x-ratelimit-limit": "60",
        "x-ratelimit-remaining": "0",
        "x-ratelimit-reset": String(Math.floor(resetMs / 1000)),
      },
      body: JSON.stringify({
        message: "API rate limit exceeded for 203.0.113.7.",
        documentation_url: "https://docs.github.com/rest/overview/rate-limits",
      }),
    });
  });
}

test("a spent budget still renders everything raw.githubusercontent can provide",
  async ({ page }) => {
    const errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    const reset = Date.now() + 45 * 60000;
    await routeExhausted(page, reset);
    await page.goto("/status.html");

    // The fleet panel is fetched from raw and needs no API call whatsoever.
    // Before this change it was fetched, resolved, and then discarded with the
    // rejected Promise.all — the section it filled stayed on "loading…".
    const fleet = page.locator("#fleet");
    await expect(fleet.locator(".card")).toHaveCount(1);
    await expect(fleet).toContainText("credit left (projected)");
    await expect(fleet).not.toContainText("loading…");

    // No section is left on the initial placeholder: the two that genuinely
    // need the API say WHY they are empty instead of pretending to load.
    for (const id of ["#runs", "#releases", "#live"]) {
      await expect(page.locator(id)).not.toContainText("loading…");
      await expect(page.locator(id)).toContainText("rate limited");
    }
    expect(errors).toEqual([]);
  });

test("the reset time is on screen, and so is the budget it ran out of",
  async ({ page }) => {
    const reset = Date.now() + 45 * 60000;
    await routeExhausted(page, reset);
    await page.goto("/status.html");

    // The banner names the clock time the page comes back, not ":09".
    const banner = page.locator("#banner");
    await expect(banner).toBeVisible();
    await expect(banner).toContainText("0 of 60 left");
    await expect(banner).toContainText("resumes " + hhmm(reset));

    // ...and the budget line states it before it runs out, every cycle.
    await expect(page.locator("#budget")).toHaveText(
      "GitHub API budget 0/60 · resets " + hhmm(reset));
    await expect(page.locator("#updated"))
      .toContainText("auto-refresh paused until " + hhmm(reset));
  });

test("the budget line reads the API's own headers on a SUCCESSFUL call too",
  async ({ page }) => {
    const reset = Date.now() + 20 * 60000;
    await page.route(/https:\/\/api\.github\.com\/.*/, async (route) => {
      const url = route.request().url();
      const body = /\/actions\/workflows\/[^/]+\/runs/.test(url) ? RUNS : [];
      await route.fulfill({
        status: 200,
        headers: {
          ...RATE_CORS,
          "x-ratelimit-limit": "60",
          "x-ratelimit-remaining": "41",
          "x-ratelimit-reset": String(Math.floor(reset / 1000)),
        },
        body: JSON.stringify(body),
      });
    });
    await page.goto("/status.html");
    await expect(page.locator("#budget")).toHaveText(
      "GitHub API budget 41/60 · resets " + hhmm(reset));
    // 41 is above the reserve, so nothing is paused and the cadence is stated.
    await expect(page.locator("#updated")).toContainText("refreshes every 5 min");
    await expect(page.locator("#banner")).toBeHidden();
  });

test("a rate-limited page renders the CACHED runs list, curves and all",
  async ({ page }) => {
    // First visit succeeds and fills the cache (the beforeEach stubs are live).
    await page.goto("/status.html");
    await expect(page.locator("#live .card").first()).toBeVisible();
    const cached = await page.evaluate(() => {
      const raw = localStorage.getItem("statusApiCache");
      return raw ? JSON.parse(raw).runs.length : 0;
    });
    expect(cached).toBe(RUNS.workflow_runs.length);

    // Now the budget is gone. Same origin, so the same localStorage.
    const reset = Date.now() + 62 * 60000;
    await routeExhausted(page, reset);
    await page.reload();

    // The table is back, from cache...
    const runs = page.locator("#runs");
    await expect(runs).toContainText("#108");
    await expect(runs).toContainText("E-017 capacity rung");
    // ...stamped with its age and the time it stops being stale, never
    // presented as live.
    await expect(runs).toContainText("runs list from");
    await expect(runs).toContainText("resumes " + hhmm(reset));

    // And the curves — which come from raw and were never rate limited —
    // are drawn for the cached runs, because the list is all they needed.
    const card = page.locator("#live .card")
      .filter({ has: page.locator("h3", { hasText: "run #102" }) });
    await expect(card).toHaveCount(1);
    await expect(card.locator("svg.chart")).toHaveCount(2);
    await expect(card).toContainText("continues run #101 from step 22,500");
  });

test("the auto poll stops when the budget is spent; the Refresh button may still spend",
  async ({ page }) => {
    // A 200 with only 3 calls left — below RATE_RESERVE (6) but not a 403, so
    // the page has real data AND knows it must stop polling. This is the case
    // that keeps the page alive: the reserve exists so the human's own button
    // still works after the timer has been told to stand down.
    const reset = Date.now() + 50 * 60000;
    const calls = [];
    await page.route(/https:\/\/api\.github\.com\/.*/, async (route) => {
      const url = route.request().url();
      calls.push(url);
      const body = /\/actions\/workflows\/[^/]+\/runs/.test(url) ? RUNS : [];
      await route.fulfill({
        status: 200,
        headers: {
          ...RATE_CORS,
          "x-ratelimit-limit": "60",
          "x-ratelimit-remaining": "3",
          "x-ratelimit-reset": String(Math.floor(reset / 1000)),
        },
        body: JSON.stringify(body),
      });
    });
    await page.goto("/status.html");
    await expect(page.locator("#live .card").first()).toBeVisible();
    const afterLoad = calls.length;
    expect(afterLoad).toBe(2);          // the runs list and /releases, once
    await expect(page.locator("#updated"))
      .toContainText("auto-refresh paused until " + hhmm(reset));

    // An automatic cycle — foregrounding the tab — must spend NOTHING.
    await page.evaluate(() =>
      document.dispatchEvent(new Event("visibilitychange")));
    await expect(page.locator("#banner")).toContainText("auto-refresh paused");
    expect(calls.length).toBe(afterLoad);

    // The button is what the reserve was held back for.
    await page.locator("#refresh").click();
    await expect.poll(() => calls.length).toBe(afterLoad + 2);
  });

test("the poll resumes at x-ratelimit-reset, not at the refresh interval",
  async ({ page }) => {
    // REFRESH_MS is 300,000 ms. A reset two seconds out proves the wait is
    // computed from the header rather than from the cadence: if the page were
    // still on setInterval(refresh, REFRESH_MS) nothing would happen here for
    // five minutes. x-ratelimit-reset is in SECONDS — a page that read it as
    // milliseconds would schedule the retry for 1970 and fire instantly, or
    // for the year 58,600 and never fire; neither passes this.
    const calls = [];
    const reset = Date.now() + 2000;
    await routeExhausted(page, reset, calls);
    await page.goto("/status.html");
    await expect(page.locator("#banner")).toBeVisible();
    expect(calls.length).toBe(2);
    // reset + 2 s of grace; allow generous slack for the sandbox.
    await expect.poll(() => calls.length, { timeout: 15000 })
      .toBeGreaterThanOrEqual(4);
  });

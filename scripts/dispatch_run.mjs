#!/usr/bin/env node
// Dispatch a training run — and REFUSE to dispatch one that has no graph.
//
// Chris has asked for this several times, most recently 2026-08-10: "every
// queued run needs a graph, this is a prerequisite to queueing and allows for
// precertification of the training run."
//
// Publishing the plan by hand after dispatching is a habit, and habits fail
// silently: #117's plan went up minutes late, #118's and #119's not at all
// until asked, and #120's went up with the wrong shape entirely (a warm
// restart drawn on a continuation's axis). CLAUDE.md's first working
// principle applies to process as much as to code — prefer the formulation
// that removes a failure mode to the one that guards against it. So the plan
// is not a step you remember; it is a required argument, and the dispatch
// cannot happen without one.
//
// PRECERTIFICATION is the second half, and the more valuable one. A plan that
// disagrees with the dispatch is worse than no plan: it will be checked
// against the live curve, it will not match, and the RUN will take the blame
// for the document. So the plan is cross-checked against the inputs before
// anything is queued — the step count, the schedule shape, the peak rate and
// the parent must all agree, or nothing is dispatched.
//
//   node scripts/dispatch_run.mjs --inputs run.json --plan plan.json \
//        --token-file ~/.gh_pat [--dry-run]
import { readFileSync } from "node:fs";

const args = process.argv.slice(2);
const arg = (k, d) => {
  const i = args.indexOf(k);
  return i >= 0 ? args[i + 1] : d;
};
const DRY = args.includes("--dry-run");
const REPO = process.env.GITHUB_REPOSITORY || "blauewelt/earth";

const inputsPath = arg("--inputs");
const planPath = arg("--plan");
if (!inputsPath || !planPath) {
  console.error(
    "usage: dispatch_run.mjs --inputs <file.json> --plan <file.json> " +
    "[--token-file <path>] [--dry-run]\n\n" +
    "--plan is REQUIRED. A queued run with no graph cannot be checked before " +
    "it spends anything, which is the whole point of publishing one.");
  process.exit(2);
}
const inputs = JSON.parse(readFileSync(inputsPath, "utf8"));
const plan = JSON.parse(readFileSync(planPath, "utf8"));
// The token is read AFTER the checks below, so --dry-run needs no credential
// and a validation failure never depends on one being present. A gate that
// only works when you are authenticated is a gate people learn to skip.

// ---- precertification ------------------------------------------------------
// Each check exists because its absence has produced a wrong or missing chart.
const problems = [];
const n = (v) => (v === undefined || v === null || v === "" ? NaN : Number(v));

// An EVAL run (sroll: with temporal_steps 0) trains nothing, so a plan with
// an LR curve is not merely unnecessary — it is FALSE, and the status page
// drew it as a real decaying schedule on #233/#294/#303 until Chris asked
// why an eval run was decaying its learning rate (2026-08-14). The honest
// plan for an eval is {"eval": true, "heads": [...]}: the page renders a
// label, not a curve. The curve checks below do not apply to it.
// `sroll:` is a window TOKEN, not necessarily the first one — every
// recipe-driven eval since the recipe mechanism landed reads
// `recipe:<name>,sroll:<head>,...`, so a startsWith test classified those
// as training dispatches and demanded an LR curve for a run with no LR
// (found 2026-08-28 dispatching E-051-roll-B; #503 had been dispatched
// around this check by publishing its plan by hand).
const isEval = String(inputs.temporal_steps ?? "") === "0"
  && String(inputs.window ?? "").split(",")
       .some((tok) => tok.startsWith("sroll:"));
// A DATA BUILD is a third kind, and the same lesson one level along: family 6
// is an unlabelled pretraining corpus, so the job builds a tensor and stops —
// the Train and Probes steps skip it. Demanding an LR curve from a run with
// no LR is what the isEval fix above was about; a build has no LR either, and
// no heads to name, so its honest plan is {"build": true, ...}.
const isBuild = String(inputs.tensor ?? "").startsWith("family6_");
if (isBuild) {
  if (!plan.build) {
    problems.push(
      "this is a BUILD dispatch (tensor family6_*) — its plan must be " +
      "{\"build\": true, ...}, not a training curve the status page would " +
      "draw as a schedule the run does not have");
  }
  if (plan.eval) {
    problems.push("plan.eval is set but this dispatch builds a tensor");
  }
} else if (plan.build) {
  problems.push("plan.build is set but this dispatch is not a tensor build");
} else if (isEval) {
  if (!plan.eval) {
    problems.push(
      "this is an EVAL dispatch (sroll:, temporal_steps 0) — its plan must " +
      "be {\"eval\": true, \"heads\": [...]}, not a training curve the " +
      "status page would draw as a schedule the run does not have");
  }
} else if (plan.eval) {
  problems.push("plan.eval is set but this dispatch trains — wrong plan file");
}

if (!isEval && !isBuild && !(n(plan.steps) > 0)) problems.push("plan.steps must be a positive number");
if (!isEval && !isBuild && !(n(plan.lr) > 0)) problems.push("plan.lr must be a positive number");
// THE CURVE MUST COME FROM THE TRAINER. A plan without `points` would be
// re-derived by the status page, which is a second implementation of the
// schedule and would happily draw a cosine for a wsd run — certifying a
// schedule the run does not use. Generate it with ml/plan_schedule.py.
if (!isEval && !isBuild && (!Array.isArray(plan.points) || plan.points.length < 2)) {
  problems.push(
    "plan.points is missing: the curve must be computed from the trainer's " +
    "own scheduler, not re-derived by the page. Generate the plan with\n" +
    "      python3 ml/plan_schedule.py --steps <n> --lr <x> --schedule <s>");
}
if (plan.schedule && String(inputs.window ?? "").startsWith("sched:")) {
  const wsched = String(inputs.window).slice("sched:".length);
  if (wsched !== plan.schedule) {
    problems.push(`window says sched:${wsched} but plan.schedule is ` +
                  `${plan.schedule} — the drawn curve is not the one that runs.`);
  }
}
if (plan.warm && (!Array.isArray(plan.parent_points) || plan.parent_points.length < 2)) {
  problems.push("a warm restart's plan needs parent_points so the chart has a " +
                "parent segment to show the seam against.");
}

const ts = String(inputs.temporal_steps ?? "");
const isJoint = ts.startsWith("joint:");
const win = String(inputs.window ?? "");
const warm = win.startsWith("warm2:");
const cont = win.startsWith("resume2:");

if (!isJoint && ts !== "0") {
  // temporal_steps is what the trainer is actually told. The plan must say
  // the same number, or the chart's x-axis is a different experiment's.
  if (n(ts) !== n(plan.steps)) {
    problems.push(
      `plan.steps (${plan.steps}) != temporal_steps (${ts}). For a warm ` +
      `restart both are the EXTRA steps; for anything else both are the total.`);
  }
}
if (warm && !plan.warm) {
  problems.push(
    "window is warm2: but plan.warm is not set — the chart would draw a " +
    "warm restart as a continuation, on the wrong axis, which is exactly " +
    "what happened to #120.");
}
if (plan.warm && !warm) {
  problems.push("plan.warm is set but window is not warm2: — the plan " +
                "describes a run that is not being dispatched.");
}
if ((warm || cont) && !(n(plan.parent_steps) > 0)) {
  problems.push("a resumed or restarted run's plan needs parent_steps, or the " +
                "chart has no seam and no parent segment to compare against.");
}
// The peak rate is carried in the window modifier as @lr; if it is there, it
// must match the plan, or the drawn schedule is not the schedule that runs.
const at = win.match(/@([0-9.eE+-]+)$/);
if (at && Math.abs(n(at[1]) - n(plan.lr)) > 1e-12) {
  problems.push(`window carries @${at[1]} but plan.lr is ${plan.lr}`);
}
if (!inputs.doc || String(inputs.doc).trim().length < 40) {
  problems.push("inputs.doc must say what this run is and why — it becomes " +
                "the run name and the status page's only description.");
}

if (problems.length) {
  console.error("REFUSING TO DISPATCH — the plan does not certify this run:\n");
  for (const p of problems) console.error("  · " + p);
  console.error("\nNothing was queued. Fix the plan or the inputs so they " +
                "describe the same experiment.");
  process.exit(1);
}
console.log("precertified: plan and inputs describe the same run");
// Say what THIS run is. The training summary printed "NaN steps · peak
// undefined" for a build, which is a log line asserting something false about
// a run that is fine — the same class of error as a plan that disagrees with
// its dispatch.
console.log(isBuild
  ? `  tensor build · ${plan.tensor ?? inputs.tensor} · ` +
    `${plan.expected?.rows?.toLocaleString() ?? "?"} rows · ` +
    `${plan.expected?.gb ?? "?"} GB · nothing trains`
  : isEval
    ? `  eval · ${(plan.heads ?? []).length} head(s) · no LR schedule`
    : `  ${plan.warm ? "warm restart" : "fresh run"} · ` +
      `${Number(plan.steps).toLocaleString()} steps · peak ${plan.lr}` +
      (plan.parent_steps
        ? ` · parent ${Number(plan.parent_steps).toLocaleString()}` : ""));
if (DRY) { console.log("--dry-run: stopping before dispatch"); process.exit(0); }

// ---- dispatch, then publish the plan under the run number it got -----------
const TOKEN = readFileSync(arg("--token-file", `${process.env.HOME}/.gh_pat`), "utf8").trim();
const H = {
  Authorization: `Bearer ${TOKEN}`,
  Accept: "application/vnd.github+json",
  "Content-Type": "application/json",
};
const before = await (await fetch(
  `https://api.github.com/repos/${REPO}/actions/workflows/ml-train.yml/runs?per_page=1`,
  { headers: H })).json();
const lastId = before.workflow_runs?.[0]?.id ?? 0;

const res = await fetch(
  `https://api.github.com/repos/${REPO}/actions/workflows/ml-train.yml/dispatches`,
  { method: "POST", headers: H, body: JSON.stringify({ ref: "main", inputs }) });
if (res.status !== 204) {
  console.error("dispatch failed:", res.status, (await res.text()).slice(0, 300));
  process.exit(1);
}
console.log("dispatched, waiting for the run number …");

let run = null;
for (let i = 0; i < 30 && !run; i++) {
  await new Promise((r) => setTimeout(r, 2000));
  const j = await (await fetch(
    `https://api.github.com/repos/${REPO}/actions/workflows/ml-train.yml/runs?per_page=3`,
    { headers: H })).json();
  run = (j.workflow_runs || []).find((r) => r.id !== lastId && r.id > lastId);
}
if (!run) {
  console.error("::warning::dispatched but could not identify the run number, " +
                "so the plan is NOT published. Publish it by hand:\n" +
                "  node scripts/publish_plan.mjs <n> '<json>'");
  process.exit(1);
}
console.log(`run #${run.run_number} (${run.id})`);

const { spawnSync } = await import("node:child_process");
const r = spawnSync("node", ["scripts/publish_plan.mjs", String(run.run_number),
                             JSON.stringify(plan)], { encoding: "utf8" });
process.stdout.write(r.stdout || "");
if (r.status !== 0) {
  console.error("::warning::the run is QUEUED but its plan failed to publish — " +
                "it has no graph. Retry:\n  node scripts/publish_plan.mjs " +
                `${run.run_number} '${JSON.stringify(plan)}'`);
  process.exit(1);
}
// Assert the EFFECT, not the invocation.
const url = `https://raw.githubusercontent.com/${REPO}/ml-metrics/plan-${run.run_number}.json`;
for (let i = 0; i < 10; i++) {
  await new Promise((r) => setTimeout(r, 1500));
  if ((await fetch(url + "?cb=" + i)).ok) {
    console.log(`plan is live and publicly readable: ${url}`);
    console.log(`https://blauewelt.github.io/earth/status.html`);
    process.exit(0);
  }
}
console.error("::warning::plan published but not yet readable at " + url);

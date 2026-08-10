#!/usr/bin/env node
// Publish a run's PLANNED schedule so the status page can draw it before the
// run has produced a single point.
//
// Chris, 2026-08-10: "we should plot the curve of the continuation run before
// it starts (incl LR) ... then we know the exact schedule." The reason is not
// curiosity. A resumed cosine loaded from a finished run gives lr = 0.0, and
// that mistake is cheapest to catch BEFORE committing sixteen hours — but the
// page could only ever draw what a run had already logged, so the schedule
// became visible only once it was too late to be worth checking.
//
// Cosine annealing is analytic, so a plan is a handful of numbers:
//
//   node scripts/publish_plan.mjs 117 '{"steps":200000,"lr":1e-4,
//        "parent_run":112,"parent_steps":60000,"parent_lr":1e-3,"at_step":60000}'
//
// Written to the ml-metrics branch, not ml-live-<n>: publish_live_metrics.sh
// force-pushes an ORPHAN commit there, so anything else on that branch is
// destroyed by the first phase update. ml-metrics is additive.
import { readFileSync } from "node:fs";

const GH = readFileSync("/home/claude/.gh_pat", "utf8").trim();
const [runNumber, planJson] = process.argv.slice(2);
if (!runNumber || !planJson) {
  console.error("usage: publish_plan.mjs <run_number> '<plan json>'");
  process.exit(2);
}
const plan = JSON.parse(planJson);
plan.published_at = new Date().toISOString();
const REPO = "blauewelt/earth", BRANCH = "ml-metrics", PATH = `plan-${runNumber}.json`;
const gh = (p, init = {}) => fetch(`https://api.github.com/repos/${REPO}${p}`, {
  ...init,
  headers: { Authorization: `Bearer ${GH}`, Accept: "application/vnd.github+json", ...(init.headers || {}) },
});
let sha;
const cur = await gh(`/contents/${PATH}?ref=${BRANCH}`);
if (cur.ok) sha = (await cur.json()).sha;
const put = await gh(`/contents/${PATH}`, {
  method: "PUT",
  body: JSON.stringify({
    message: `plan for run ${runNumber}`,
    content: Buffer.from(JSON.stringify(plan, null, 1) + "\n").toString("base64"),
    branch: BRANCH, ...(sha ? { sha } : {}),
  }),
});
console.log(put.ok ? `published ${BRANCH}/${PATH}`
                   : `FAILED ${put.status}: ${(await put.text()).slice(0, 200)}`);

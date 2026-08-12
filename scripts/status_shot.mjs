#!/usr/bin/env node
// Screenshot the status page AS THE USER SEES IT, from live data.
//
// Chris, 2026-08-10: "Maybe you can check/screenshot the status page during
// monitoring wake-up yourself in the future? Make it a standing rule."
//
// The reason is the day's running theme. Twice I told him the dashboard showed
// something it did not — once because the plan preview rendered on a branch
// nobody is ever on, once because a stale record from a previous run was being
// charted as the current one. Both times I was reporting what the code should
// produce rather than what the page produces. A screenshot is the only check
// that cannot be fooled by my model of the code.
//
// The sandbox browser cannot reach the network, so this captures the real
// inputs with fetch (which can), stubs them into the page, and renders. That
// is not a simulation: the page is the deployed file and the data is the
// running jobs'.
//
//   node scripts/status_shot.mjs [--out /tmp/status.png] [--runs 6]
import { readFileSync, writeFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawn } from "node:child_process";

const args = process.argv.slice(2);
const arg = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const OUT = arg("--out", "/tmp/status.png");
// MATCH THE PAGE — status.html's RUN_WINDOW, which is 30. Capturing fewer
// makes the screenshot show LESS than the user sees, which defeats its
// purpose. This has now hidden the SAME live experiment twice: first at 5,
// where #121 was simply outside the capture, and then at 12, which matched
// the page but only because the page had the identical defect — eight
// dispatches in twenty minutes pushed a six-hour run out of both windows at
// once, and the screenshot faithfully reproduced its absence.
const N = Number(arg("--runs", "30"));
const REPO = process.env.GITHUB_REPOSITORY || "blauewelt/earth";
const TOKEN = (() => {
  // Default to the session's token location, not $HOME — this runs as root in
  // the sandbox, where $HOME/.gh_pat does not exist and an unauthenticated
  // list gets a 403 that looks like a permissions problem rather than a
  // missing file.
  for (const p of [arg("--token-file"), "/home/claude/.gh_pat",
                   `${process.env.HOME}/.gh_pat`]) {
    if (!p) continue;
    try { return readFileSync(p, "utf8").trim(); } catch { /* next */ }
  }
  return "";
})();

const H = TOKEN ? { Authorization: `Bearer ${TOKEN}`, Accept: "application/vnd.github+json" } : {};
const txt = async (u) => { try { const r = await fetch(u); return r.ok ? await r.text() : null; } catch { return null; } };

const runsRes = await fetch(
  `https://api.github.com/repos/${REPO}/actions/workflows/ml-train.yml/runs?per_page=${N}`, { headers: H });
if (!runsRes.ok) { console.error("cannot list runs:", runsRes.status); process.exit(1); }
const RUNS = { workflow_runs: (await runsRes.json()).workflow_runs };

// Releases must be real: the stub used to answer "[]" for every API URL
// except the runs list, so the RELEASES section rendered "no releases
// found" in every screenshot — the one check that exists so my model of
// the page cannot fool me was itself serving fiction for that section
// (caught 2026-08-12 when it matched a transient on Chris's phone).
const relRes = await fetch(`https://api.github.com/repos/${REPO}/releases`, { headers: H });
const RELS = relRes.ok ? await relRes.json() : [];

const data = { runs: RUNS, rels: RELS, live: {}, phase: {}, plan: {}, docs: "{}", fleet: null };
for (const r of RUNS.workflow_runs) {
  const n = r.run_number;
  data.live[n] = await txt(`https://raw.githubusercontent.com/${REPO}/ml-live-${n}/metrics.jsonl`)
              ?? await txt(`https://raw.githubusercontent.com/${REPO}/ml-metrics/run-${n}.jsonl`);
  data.phase[n] = await txt(`https://raw.githubusercontent.com/${REPO}/ml-live-${n}/phase.json`);
  data.plan[n] = await txt(`https://raw.githubusercontent.com/${REPO}/ml-metrics/plan-${n}.json`);
}
data.docs = await txt(`https://raw.githubusercontent.com/${REPO}/main/ml/run_docs.json`) ?? "{}";
data.fleet = await txt(`https://raw.githubusercontent.com/${REPO}/ml-metrics/fleet.json`);
const have = Object.entries(data.live).filter(([, v]) => v).map(([k]) => "#" + k);
console.log(`captured ${RUNS.workflow_runs.length} runs; metrics for ${have.join(", ") || "none"}`);

const dir = mkdtempSync(join(tmpdir(), "shot-"));
writeFileSync(join(dir, "data.json"), JSON.stringify(data));
writeFileSync(join(dir, "shot.mjs"), `
// playwright is CJS; a named ESM import of it is fragile across versions.
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
const require = createRequire(${JSON.stringify("file://" + process.cwd() + "/")});
const { chromium } = require("playwright");
const D = JSON.parse(readFileSync(${JSON.stringify(join(dir, "data.json"))}, "utf8"));
const b = await chromium.launch();
// A phone viewport, because that is where this gets read.
const p = await b.newPage({ viewport: { width: 430, height: 1600 }, deviceScaleFactor: 2 });
const errs = [];
p.on("pageerror", (e) => errs.push(e.message));
await p.route(/https:\\/\\/api\\.github\\.com\\/.*/, (r) => {
  const u = r.request().url();
  const body = /workflows/.test(u) ? JSON.stringify(D.runs)
             : /releases/.test(u)  ? JSON.stringify(D.rels)
             : "[]";
  return r.fulfill({ status: 200, contentType: "application/json", body });
});
await p.route(/https:\\/\\/raw\\.githubusercontent\\.com\\/.*/, (r) => {
  const u = r.request().url();
  const ok = (t) => r.fulfill({ status: 200, contentType: "text/plain", body: t });
  let m;
  if (/run_docs\\.json/.test(u)) return ok(D.docs);
  if (/fleet\\.json/.test(u)) return D.fleet ? ok(D.fleet) : r.fulfill({ status: 404, body: "" });
  if ((m = u.match(/ml-live-(\\d+)\\/metrics/))) return D.live[m[1]] ? ok(D.live[m[1]]) : r.fulfill({ status: 404, body: "" });
  if ((m = u.match(/ml-live-(\\d+)\\/phase/))) return D.phase[m[1]] ? ok(D.phase[m[1]]) : r.fulfill({ status: 404, body: "" });
  if ((m = u.match(/plan-(\\d+)\\.json/))) return D.plan[m[1]] ? ok(D.plan[m[1]]) : r.fulfill({ status: 404, body: "" });
  if ((m = u.match(/ml-metrics\\/run-(\\d+)\\.jsonl/))) return D.live[m[1]] ? ok(D.live[m[1]]) : r.fulfill({ status: 404, body: "" });
  return r.fulfill({ status: 404, body: "" });
});
await p.goto("http://localhost:8099/status.html", { waitUntil: "load" });
await p.waitForTimeout(2500);
await p.screenshot({ path: ${JSON.stringify(OUT)}, fullPage: true });
console.log("PAGE ERRORS: " + (errs.length ? errs.join(" | ") : "none"));
await b.close();
`);

const srv = spawn("python3", ["-m", "http.server", "8099"], { cwd: process.cwd(), stdio: "ignore" });
await new Promise((r) => setTimeout(r, 1200));
const sh = spawn("node", [join(dir, "shot.mjs")], { cwd: process.cwd(), stdio: "inherit" });
const code = await new Promise((r) => sh.on("close", r));
srv.kill();
if (code === 0) console.log("wrote " + OUT);
process.exit(code);

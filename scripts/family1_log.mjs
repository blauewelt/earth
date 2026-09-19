// Tail of a family1-build run's log, by RUN NUMBER (the #NNN the runs list
// prints), so a failed lane can be read without a desktop browser.
//   node scripts/family1_log.mjs <run_number> [--lines 60] [--grep regex]
// PAT from /home/claude/.gh_pat — file read only, never argv.
import fs from "node:fs";
const args = process.argv.slice(2);
const num = args[0];
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const lines = Number(opt("--lines", "60")), grep = opt("--grep", "");
const tok = fs.readFileSync("/home/claude/.gh_pat", "utf8").trim();
const h = { Authorization: `Bearer ${tok}`, Accept: "application/vnd.github+json" };
const R = "https://api.github.com/repos/blauewelt/earth";
const runs = await (await fetch(`${R}/actions/workflows/family1-build.yml/runs?per_page=100`, { headers: h })).json();
const run = (runs.workflow_runs ?? []).find(w => String(w.run_number) === String(num));
if (!run) { console.error(`run #${num} not in the last 100`); process.exit(2); }
console.log(`#${run.run_number} ${run.conclusion ?? run.status} ${run.name} (${run.id})`);
const jobs = await (await fetch(`${R}/actions/runs/${run.id}/jobs`, { headers: h })).json();
for (const job of jobs.jobs ?? []) {
  console.log(`job ${job.name}: ${job.conclusion ?? job.status} runner=${job.runner_name ?? "-"} started=${job.started_at}`);
  for (const s of job.steps ?? []) if (s.conclusion && s.conclusion !== "success" && s.conclusion !== "skipped") console.log(`  step failed: ${s.name}`);
  const r = await fetch(`${R}/actions/jobs/${job.id}/logs`, { headers: h, redirect: "follow" });
  if (r.status !== 200) { console.log(`  log: HTTP ${r.status} (job still running?)`); continue; }
  let txt = (await r.text()).split("\n").map(l => l.replace(/^\S+T\S+Z /, ""));
  if (grep) txt = txt.filter(l => new RegExp(grep).test(l));
  console.log(txt.slice(-lines).join("\n"));
}

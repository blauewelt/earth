#!/usr/bin/env node
// Is the fleet actually working? One command, one verdict, no interpretation.
//
// Written after 2026-08-10, when a job sat for eight hours at gpu_util=0 and
// cpu_util=60% because four scripts in the probe ladder had never been moved
// off the CPU. Nothing was broken in a way anything reported: the job was
// "in progress", the runner was "online", the box was "running". The only
// signal that would have caught it is the one nobody was looking at.
//
//   node scripts/fleet_health.mjs          # human summary + exit 1 if unhealthy
//
// Checks, in the order they bite:
//   1. a box RUNNING (paying) with NO job on it        -> idle burn
//   2. a job running with the GPU at ~0                -> CPU-bound work
//   3. a job queued while a runner is online and idle  -> the Actions stall
//   4. disk over 90%                                   -> takes a runner offline
import { readFileSync } from "node:fs";

const VAST = readFileSync("/home/claude/.vast_key", "utf8").trim();
const GH = readFileSync("/home/claude/.gh_pat", "utf8").trim();
const gh = (p) => fetch(`https://api.github.com/repos/blauewelt/earth${p}`, {
  headers: { Authorization: `Bearer ${GH}`, Accept: "application/vnd.github+json" },
}).then((r) => r.json());
const vast = (p) => fetch(`https://console.vast.ai/api/v1${p}`, {
  headers: { Authorization: `Bearer ${VAST}` },
}).then((r) => r.json());

const [inst, runners, runs] = await Promise.all([
  vast("/instances/"), gh("/actions/runners"),
  gh("/actions/workflows/ml-train.yml/runs?per_page=15"),
]);

// Runner names are stale instance IDs; the only mapping is inside onstart.
const nameOf = new Map((inst.instances || []).map((i) => {
  const m = (i.onstart || "").match(/--name[= ]"?([^"\s\\]+)/);
  return [i.id, m ? m[1] : String(i.id)];
}));
const live = (runs.workflow_runs || []).filter((r) => r.status !== "completed");
const jobs = await Promise.all(live.map((r) =>
  gh(`/actions/runs/${r.id}/jobs`).then((j) => ({ run: r, job: (j.jobs || [])[0] }))));
const busyRunners = new Set(jobs.filter((x) => x.job?.runner_name).map((x) => x.job.runner_name));

const problems = [];
for (const i of inst.instances || []) {
  if (i.actual_status !== "running") continue;
  const name = nameOf.get(i.id);
  const running = busyRunners.has(name);
  const gpu = Number(i.gpu_util || 0), cpu = Number(i.cpu_util || 0);
  const diskPct = 100 * (i.disk_util || 0) / (i.disk_space || 1);
  const tag = `${name} (${i.id}, $${(i.dph_total || 0).toFixed(3)}/h)`;
  if (!running) problems.push(`IDLE BURN   ${tag} is running with no job on it`);
  else if (gpu < 5 && cpu > 20)
    problems.push(`CPU-BOUND   ${tag} has a job but gpu_util=${gpu}% cpu_util=${cpu.toFixed(0)}%`);
  if (diskPct > 90) problems.push(`DISK        ${tag} at ${diskPct.toFixed(0)}% — a full disk takes a runner offline`);
  console.log(`  ${tag.padEnd(46)} job=${running ? "yes" : "NO "} gpu=${String(gpu).padStart(3)}% cpu=${cpu.toFixed(0).padStart(3)}% disk=${diskPct.toFixed(0)}%`);
}
const idleOnline = (runners.runners || []).filter((r) => r.status === "online" && !r.busy);
for (const { run, job } of jobs) {
  if (job?.status === "queued" && idleOnline.length)
    problems.push(`QUEUE STALL #${run.run_number} is queued while ${idleOnline.map((r) => r.name).join(", ")} sits online and idle — cancel and re-dispatch`);
}
console.log(`\n${live.length} run(s) not finished · ${idleOnline.length} runner(s) online+idle`);
if (!problems.length) { console.log("HEALTHY"); process.exit(0); }
console.log("\n" + problems.map((p) => "  ! " + p).join("\n"));
process.exit(1);

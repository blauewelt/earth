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

// per_page must cover EVERY live run, not a taste of them. At 15, the
// 2026-08-14 E-026 fan-out (10 boxes, 22 live runs) fell off the window:
// six busy boxes read as IDLE BURN (their runs were older than the 15
// newest) and this script printed "stop it" for boxes mid-job. 100 is the
// API max; live runs are bounded by fleet size + queue depth, far below it.
const [inst, runners, runs] = await Promise.all([
  vast("/instances/"), gh("/actions/runners?per_page=100"),
  gh("/actions/workflows/ml-train.yml/runs?per_page=100"),
]);

// DEAD TELEMETRY FRAMES (2026-08-15, run #310): /instances/ intermittently
// returns a GPU frame of ALL zeros — gpu_util 0, gpu_temp 0, vmem_usage 0 —
// for a box mid-training (live frames ~20s apart read 89-100% util, 69-75°C,
// 16 GB VRAM). cpu_util is collected host-side and stays real, so a dead
// frame walks straight into the CPU-BOUND condition and a box training at
// full speed reads as the §2f failure. A REAL cpu-bound box still produces a
// live frame — an idle GPU reads ~30-50°C, never exactly 0°C — so a frame
// with temp AND vmem both 0 carries no information. Resample until a live
// frame appears (they alternate within ~20s), and only judge gpu_util on a
// live frame. Ground truth if in doubt: the run's ml-live-<n> branch — a
// training job pushes stage2_wall_s there every ~5 min.
const deadFrame = (i) => !Number(i.gpu_temp || 0) && !Number(i.vmem_usage || 0);
let instances = inst.instances || [];
for (let tries = 0;
     tries < 3 && instances.some((i) => i.actual_status === "running" && deadFrame(i));
     tries++) {
  await new Promise((r) => setTimeout(r, 20000));
  const again = await vast("/instances/");
  const fresh = new Map((again.instances || []).map((i) => [i.id, i]));
  instances = instances.map((i) =>
    deadFrame(i) && fresh.has(i.id) && !deadFrame(fresh.get(i.id)) ? fresh.get(i.id) : i);
}

// Runner names are stale instance IDs; the only mapping is inside onstart.
const nameOf = new Map(instances.map((i) => {
  const m = (i.onstart || "").match(/--name[= ]"?([^"\s\\]+)/);
  return [i.id, m ? m[1] : String(i.id)];
}));
const live = (runs.workflow_runs || []).filter((r) => r.status !== "completed");
const jobs = await Promise.all(live.map((r) =>
  gh(`/actions/runs/${r.id}/jobs`).then((j) => ({ run: r, job: (j.jobs || [])[0] }))));
const busyRunners = new Set(jobs.filter((x) => x.job?.runner_name).map((x) => x.job.runner_name));

const problems = [];
for (const i of instances) {
  if (i.actual_status !== "running") continue;
  const name = nameOf.get(i.id);
  const running = busyRunners.has(name);
  const gpu = Number(i.gpu_util || 0), cpu = Number(i.cpu_util || 0);
  const diskPct = 100 * (i.disk_util || 0) / (i.disk_space || 1);
  const tag = `${name} (${i.id}, $${(i.dph_total || 0).toFixed(3)}/h)`;
  if (!running) problems.push(`IDLE BURN   ${tag} is running with no job on it`);
  else if (deadFrame(i))
    problems.push(`TELEMETRY   ${tag} GPU stats read all-zero for ~60s — CPU-BOUND check is blind; verify via the run's ml-live branch before acting`);
  else if (gpu < 5 && cpu > 20)
    problems.push(`CPU-BOUND   ${tag} has a job but gpu_util=${gpu}% cpu_util=${cpu.toFixed(0)}%`);
  if (diskPct > 90) problems.push(`DISK        ${tag} at ${diskPct.toFixed(0)}% — a full disk takes a runner offline`);
  console.log(`  ${tag.padEnd(46)} job=${running ? "yes" : "NO "} gpu=${String(gpu).padStart(3)}% cpu=${cpu.toFixed(0).padStart(3)}% disk=${diskPct.toFixed(0)}%`);
}
const idleOnline = (runners.runners || []).filter((r) => r.status === "online" && !r.busy);
const idleNames = new Set(idleOnline.map((r) => r.name));
for (const { run, job } of jobs) {
  if (job?.status !== "queued") continue;
  // Dispatches pin runs to one runner via a gpu-box-* label; a run queued
  // behind ITS OWN busy box is the normal fan-out pattern, not a stall
  // (the documented label-pin false positive). A stall is: the runner this
  // run is pinned to (or, unpinned, any runner) sits idle — and has for a
  // few minutes, because a just-freed box takes ~1 min to pick up its next
  // pinned run (watched #278 do exactly that, 2026-08-14 17:23→17:28).
  const pins = (job.labels || []).filter((l) => l.startsWith("gpu-box-"));
  const target = pins.length ? pins.filter((p) => idleNames.has(p)) : [...idleNames];
  const ageMin = (Date.now() - new Date(run.created_at)) / 60000;
  if (target.length && ageMin > 5)
    problems.push(`QUEUE STALL #${run.run_number} queued ${ageMin.toFixed(0)} min while ${target.join(", ")} sits online and idle — cancel and re-dispatch`);
}
console.log(`\n${live.length} run(s) not finished · ${idleOnline.length} runner(s) online+idle`);
if (!problems.length) { console.log("HEALTHY"); process.exit(0); }
console.log("\n" + problems.map((p) => "  ! " + p).join("\n"));
process.exit(1);

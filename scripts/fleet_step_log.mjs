// GitHub's per-job log endpoint only serves the FULL log once a job ends.
// While it runs, the per-STEP log is available through the (undocumented but
// stable) logs endpoint with the step index — this pulls one step's text so
// we can read "torch ... cuda: True" before the job finishes.
import { readFileSync } from "node:fs";
const GH = readFileSync("/home/claude/.gh_pat", "utf8").trim();
const h = { Authorization: `Bearer ${GH}`, Accept: "application/vnd.github+json" };
const [runId, want] = process.argv.slice(2);
const j = await (await fetch(
  `https://api.github.com/repos/blauewelt/earth/actions/runs/${runId}/jobs`, { headers: h })).json();
const job = j.jobs[0];
const res = await fetch(
  `https://api.github.com/repos/blauewelt/earth/actions/jobs/${job.id}/logs`,
  { headers: h, redirect: "follow" });
if (!res.ok) { console.log(`logs -> ${res.status} (job still running; retry after it ends)`); process.exit(0); }
const txt = await res.text();
const re = new RegExp(want ?? "cuda", "i");
const hits = txt.split("\n").filter((l) => re.test(l));
console.log(hits.slice(-20).join("\n") || "(no matching lines yet)");

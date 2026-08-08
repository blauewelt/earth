import { readFileSync } from "node:fs";
const GH = readFileSync("/home/claude/.gh_pat", "utf8").trim();
const h = { Authorization: `Bearer ${GH}`, Accept: "application/vnd.github+json" };
const r = await (await fetch(
  `https://api.github.com/repos/blauewelt/earth/actions/runs/${process.argv[2]}`,
  { headers: h })).json();
const j = await (await fetch(
  `https://api.github.com/repos/blauewelt/earth/actions/runs/${process.argv[2]}/jobs`,
  { headers: h })).json();
const job = (j.jobs ?? [])[0] ?? {};
const cur = (job.steps ?? []).find((s) => s.status === "in_progress");
const failed = (job.steps ?? []).filter((s) => s.conclusion === "failure").map((s) => s.name);
console.log(`#${r.run_number} ${r.status} ${r.conclusion ?? ""} step=${cur ? cur.name : "-"}` +
  (failed.length ? ` FAILED=[${failed.join("; ")}]` : ""));

import { readFileSync } from "node:fs";
const TOKEN = readFileSync("/home/claude/.gh_pat", "utf8").trim();
const r = await fetch(
  "https://api.github.com/repos/blauewelt/earth/actions/workflows/ml-train.yml/runs?per_page=6",
  { headers: { Authorization: `Bearer ${TOKEN}` } });
for (const run of (await r.json()).workflow_runs)
  console.log("#" + run.run_number, run.head_branch, run.status,
              run.conclusion ?? "", run.created_at, run.id);

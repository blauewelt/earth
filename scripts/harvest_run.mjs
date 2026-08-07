#!/usr/bin/env node
// Pull an ml-train run's artifacts off Actions and into ml/runs/<name>/.
//
// This exists because the harvest has been hand-written into /tmp three times
// and lost to a container restart all three (2026-08-07). The standardised
// suite is only standard if it is in the repo.
//
// Usage:
//   node scripts/harvest_run.mjs --run-number 14 --name global25
//   node scripts/harvest_run.mjs --run-number 14 --name global25 --no-probes
//
// Token from a FILE, never argv (CLAUDE.md §1):  --token-file /home/claude/.gh_pat
//
// Artifacts written by .github/workflows/ml-train.yml:
//   pixelmae-<n>  pixelmae.pt · eval.json · metrics.jsonl · curve.png
//   probes-<n>    probe_sequence.json · temporal.json · temporal.pt
//
// NOTE ON temporal.pt: a stage-2 trained on a runner BEFORE the codec-aware
// embed-cache fix (main 5662376) may have been trained on ANOTHER run's
// embeddings — the poisoned-cache bug. --no-probes skips it; when in doubt
// retrain stage 2 locally, which is what the suite below does anyway.
import { execFileSync } from "node:child_process";
import { mkdirSync, writeFileSync, readFileSync, existsSync } from "node:fs";
import { join } from "node:path";

const arg = (n, d) => { const i = process.argv.indexOf(n); return i > 0 ? process.argv[i + 1] : d; };
const REPO = arg("--repo", "blauewelt/earth");
const NUM = Number(arg("--run-number", "0"));
const NAME = arg("--name", "");
const TOKEN = readFileSync(arg("--token-file", "/home/claude/.gh_pat"), "utf8").trim();
if (!NUM || !NAME) { console.error("--run-number and --name are required"); process.exit(2); }

const api = async (path) => {
  const r = await fetch(`https://api.github.com/repos/${REPO}${path}`, {
    headers: { Authorization: `Bearer ${TOKEN}`, Accept: "application/vnd.github+json" } });
  if (!r.ok) throw new Error(`${path} -> ${r.status}: ${(await r.text()).slice(0, 200)}`);
  return r.json();
};

const runs = await api(`/actions/workflows/ml-train.yml/runs?per_page=30`);
const run = runs.workflow_runs.find((r) => r.run_number === NUM);
if (!run) { console.error(`no ml-train run #${NUM}`); process.exit(1); }
console.log(`#${NUM} ${run.head_branch} ${run.status} ${run.conclusion ?? ""} ${run.created_at}`);
if (run.status !== "completed")
  console.log("  (still running — artifacts appear only when the job ends)");

const dest = join("ml", "runs", NAME);
mkdirSync(dest, { recursive: true });
const arts = await api(`/actions/runs/${run.id}/artifacts`);
const want = process.argv.includes("--no-probes")
  ? [`pixelmae-${NUM}`] : [`pixelmae-${NUM}`, `probes-${NUM}`];

for (const a of arts.artifacts.filter((a) => want.includes(a.name))) {
  const res = await fetch(a.archive_download_url, {
    headers: { Authorization: `Bearer ${TOKEN}` }, redirect: "follow" });
  if (!res.ok) { console.error(`  ${a.name}: HTTP ${res.status}`); continue; }
  const zip = join("/tmp", `${a.name}.zip`);
  writeFileSync(zip, Buffer.from(await res.arrayBuffer()));
  // -j: the workflow uploads with the ml/runs/actions/ prefix intact
  execFileSync("unzip", ["-o", "-j", zip, "-d", dest], { stdio: "inherit" });
  console.log(`  ${a.name} -> ${dest} (${(a.size_in_bytes / 1e6).toFixed(1)} MB)`);
}

if (!existsSync(join(dest, "pixelmae.pt"))) {
  console.error("\nno pixelmae.pt — the job produced no checkpoint (timeout without " +
                "--max-minutes?). Nothing to harvest.");
  process.exit(1);
}

console.log(`
next — the standardised suite (the tensor must MATCH the codec's channels):
  python3 ml/trainprobe.py   --run ${NAME}
  python3 ml/probe_kfold.py  --runs ${NAME}
  python3 ml/dip_check.py    --run ${NAME}
  python3 ml/temporal.py     --run ${NAME} --K 24 --steps 6000   # stage 2, locally
  python3 ml/rollout.py      --run ${NAME}
`);

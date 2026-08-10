#!/usr/bin/env node
// Pull a run's probe_kfold.json out of its Actions artifact and merge it into
// the run's archived bundle on ml-metrics.
//
// Why this exists. probe_kfold.py writes `ml/runs/probe_kfold.json`, one level
// above the per-run directory, and archive_probes.mjs was called with
// `--dir ml/runs/actions` — so bundles archived before 2026-08-10 20:5x carry
// the head numbers without the CODEC CONTROL they are read against. The
// archiver is fixed, but E-009's four arms were already queued against the
// older build when the fix landed, and re-dispatching a third time to gain one
// file is more churn than it is worth.
//
// The alternative was "remember to do it by hand afterwards", and this repo
// has a standing result about that: publishing the plan by hand was a habit,
// and habits fail silently — #117's went up late, #118's and #119's not at
// all. So the recovery is a command.
//
// It is also the general repair for any run whose bundle predates the fix,
// while the artifact is still inside its 30-day window.
//
//   node scripts/rescue_probe_kfold.mjs --runs 131,132,133,134 [--token-file ~/.gh_pat]
import { readFileSync, writeFileSync, mkdtempSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { execFileSync } from "node:child_process";

const args = process.argv.slice(2);
const arg = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const REPO = process.env.GITHUB_REPOSITORY || "blauewelt/earth";
const RUNS = (arg("--runs", "") || "").split(",").filter(Boolean).map(Number);
if (!RUNS.length) {
  console.error("usage: rescue_probe_kfold.mjs --runs 131,132 [--token-file <path>]");
  process.exit(2);
}
const TOKEN = (() => {
  for (const p of [arg("--token-file"), "/home/claude/.gh_pat", `${process.env.HOME}/.gh_pat`]) {
    if (!p) continue;
    try { const t = readFileSync(p, "utf8").trim(); if (t) return t; } catch { /* next */ }
  }
  return (process.env.GITHUB_TOKEN || "").trim();
})();
if (!TOKEN) { console.error("no token"); process.exit(1); }
const H = { Authorization: `Bearer ${TOKEN}`, Accept: "application/vnd.github+json" };
const HW = { Authorization: `token ${TOKEN}`, Accept: "application/vnd.github+json",
             "Content-Type": "application/json" };

const runs = await (await fetch(
  `https://api.github.com/repos/${REPO}/actions/workflows/ml-train.yml/runs?per_page=60`,
  { headers: H })).json();

let fixed = 0, skipped = 0;
for (const n of RUNS) {
  const run = runs.workflow_runs.find((r) => r.run_number === n);
  if (!run) { console.log(`#${n}: not in the last 60 runs`); skipped++; continue; }

  // Is it already there? Read through the API, never raw — the CDN copy lags
  // minutes behind a write and would make this re-upload what it just wrote.
  const cur = await fetch(
    `https://api.github.com/repos/${REPO}/contents/probes-${n}.json?ref=ml-metrics`,
    { headers: H });
  if (!cur.ok) { console.log(`#${n}: no bundle on ml-metrics yet`); skipped++; continue; }
  const meta = await cur.json();
  const bundle = JSON.parse(Buffer.from(meta.content, "base64").toString("utf8"));
  if (bundle.files?.["probe_kfold.json"]) {
    console.log(`#${n}: already has the codec control — nothing to do`);
    skipped++; continue;
  }

  const arts = await (await fetch(
    `https://api.github.com/repos/${REPO}/actions/runs/${run.id}/artifacts`,
    { headers: H })).json();
  const art = (arts.artifacts || []).find((a) => a.name === `probes-${n}`);
  if (!art) { console.log(`#${n}: no probes artifact (expired or never uploaded)`); skipped++; continue; }
  if (art.expired) { console.log(`#${n}: artifact EXPIRED — the control is gone`); skipped++; continue; }

  const dir = mkdtempSync(join(tmpdir(), `rescue-${n}-`));
  const zip = join(dir, "a.zip");
  const r = await fetch(
    `https://api.github.com/repos/${REPO}/actions/artifacts/${art.id}/zip`, { headers: H });
  if (!r.ok) { console.log(`#${n}: artifact download ${r.status}`); skipped++; continue; }
  writeFileSync(zip, Buffer.from(await r.arrayBuffer()));
  execFileSync("unzip", ["-o", "-q", zip, "-d", dir]);
  const p = join(dir, "probe_kfold.json");
  if (!existsSync(p)) { console.log(`#${n}: artifact has no probe_kfold.json`); skipped++; continue; }

  bundle.files["probe_kfold.json"] = JSON.parse(readFileSync(p, "utf8"));
  bundle.rescued_probe_kfold_at = new Date().toISOString();
  const put = await fetch(
    `https://api.github.com/repos/${REPO}/contents/probes-${n}.json`,
    { method: "PUT", headers: HW, body: JSON.stringify({
        message: `merge the codec control into run #${n}'s probe bundle`,
        content: Buffer.from(JSON.stringify(bundle, null, 1)).toString("base64"),
        branch: "ml-metrics", sha: meta.sha }) });
  if (!put.ok) { console.log(`#${n}: write failed ${put.status}`); skipped++; continue; }

  // Assert the effect, through the API for the same staleness reason.
  const back = await fetch(
    `https://api.github.com/repos/${REPO}/contents/probes-${n}.json?ref=ml-metrics`,
    { headers: H });
  const ok = back.ok && JSON.parse(Buffer.from((await back.json()).content, "base64")
                                     .toString("utf8")).files["probe_kfold.json"];
  console.log(`#${n}: ${ok ? "codec control merged" : "WROTE IT BUT IT IS NOT THERE"}`);
  if (ok) fixed++; else skipped++;
}
console.log(`\n${fixed} bundle(s) repaired, ${skipped} skipped.`);

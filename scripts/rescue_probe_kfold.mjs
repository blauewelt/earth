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
import { readFileSync, writeFileSync, mkdtempSync, existsSync, readdirSync } from "node:fs";
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
  // TWO CASES, not one. A run dispatched before 508e717 (2026-08-10 19:23)
  // never called archive_probes at all, so there is no bundle to repair —
  // #116, #125 and #126 are all in that window, and the first two were
  // rebuilt by hand tonight before this tool existed. A run dispatched after
  // it but before a70f3a41 has a bundle missing only the codec control.
  // Both end with the same artifact and the same upload, so both live here.
  let meta = null, bundle = null;
  if (cur.ok) {
    meta = await cur.json();
    bundle = JSON.parse(Buffer.from(meta.content, "base64").toString("utf8"));
    if (bundle.files?.["probe_kfold.json"]) {
      console.log(`#${n}: already has the codec control — nothing to do`);
      skipped++; continue;
    }
    console.log(`#${n}: bundle exists, missing the codec control`);
  } else if (cur.status === 404) {
    console.log(`#${n}: no bundle at all — building one from the artifact`);
    bundle = { run_number: n, files: {} };
  } else {
    console.log(`#${n}: cannot read the bundle (${cur.status})`); skipped++; continue;
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
  // The artifact keeps the on-box layout: probe_kfold.json at the top,
  // everything else under actions/. Take every result file from wherever it
  // sits — for a bundle being built from scratch the head numbers matter more
  // than the control, and for one being repaired the files it already has are
  // left alone.
  const WANT = ["probe_kfold.json", "temporal.json", "probe_sequence.json",
                "dip_check.json", "probe_head.json", "probe_head_raw3x3.json",
                "probe_head_raw.json", "rollout.json", "provenance.json"];
  // ...plus EVERY OTHER probe_head*.json actually in the artifact. Since
  // 2026-08-21 that family is generated rather than enumerated: --target adds
  // a suffix per transport series and --raw --wind-only adds the head's own
  // unpooled BAR (ml/probe_head.py). Rescuing the head and dropping its
  // matched bar would leave a bundle whose only available bar is the POOLED
  // one, and scripts/sweep_table.mjs then correctly refuses to draw a
  // verdict — a repair that quietly costs the comparison it was made for.
  // Mirrors scripts/archive_probes.py's WANT_GLOB.
  const found = new Set(WANT);
  for (const base of [dir, join(dir, "actions")]) {
    if (!existsSync(base)) continue;
    for (const f of readdirSync(base)) {
      if (/^probe_head.*\.json$/.test(f)) found.add(f);
    }
  }
  let added = 0;
  for (const f of found) {
    const q = [join(dir, f), join(dir, "actions", f)].find((x) => existsSync(x));
    if (!q || bundle.files[f]) continue;
    try { bundle.files[f] = JSON.parse(readFileSync(q, "utf8")); added++; }
    catch (e) { console.log(`  #${n}: skipping ${f} (${e.message})`); }
  }
  if (!added) { console.log(`#${n}: artifact had nothing new`); skipped++; continue; }
  console.log(`  #${n}: adding ${added} file(s); bundle will hold ` +
              WANT.filter((f) => bundle.files[f]).join(", "));
  if (!bundle.files["probe_kfold.json"]) {
    console.log(`  ::warning:: #${n} still has no codec control — the artifact ` +
                `did not carry probe_kfold.json`);
  }
  bundle.rescued_at = new Date().toISOString();
  const put = await fetch(
    `https://api.github.com/repos/${REPO}/contents/probes-${n}.json`,
    { method: "PUT", headers: HW, body: JSON.stringify({
        message: meta ? `merge the codec control into run #${n}'s probe bundle`
                      : `archive run #${n}'s probe results from its artifact`,
        content: Buffer.from(JSON.stringify(bundle, null, 1)).toString("base64"),
        branch: "ml-metrics", ...(meta ? { sha: meta.sha } : {}) }) });
  if (!put.ok) { console.log(`#${n}: write failed ${put.status}`); skipped++; continue; }

  // Assert the effect, through the API for the same staleness reason — but
  // RETRY. A file CREATED a moment ago can still read 404 on the contents
  // endpoint for a second or two, and the first version checked exactly once:
  // #126's bundle landed complete with all four files and the tool printed
  // "WROTE IT BUT IT IS NOT THERE". A verification that cries wolf is worse
  // than none, because the next false alarm gets ignored along with the true
  // one. archive_probes.mjs already retries for this reason; this did not.
  let got = null;
  for (let i = 0; i < 6 && !got; i++) {
    if (i) await new Promise((r) => setTimeout(r, 1200));
    const back = await fetch(
      `https://api.github.com/repos/${REPO}/contents/probes-${n}.json?ref=ml-metrics`,
      { headers: H });
    if (!back.ok) continue;
    const body = await back.json();
    if (!body.content) continue;
    const files = JSON.parse(Buffer.from(body.content, "base64").toString("utf8")).files;
    if (files && Object.keys(files).length) got = files;
  }
  console.log(`#${n}: ${got ? "archived — " + Object.keys(got).join(", ")
                            : "WROTE IT BUT IT IS STILL NOT READABLE after 6 tries"}`);
  if (got) fixed++; else skipped++;
}
console.log(`\n${fixed} bundle(s) repaired, ${skipped} skipped.`);

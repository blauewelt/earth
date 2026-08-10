#!/usr/bin/env node
// Tabulate a SWEEP from the archived probe bundles on ml-metrics.
//
// Written while E-009's four arms were queued, so that reading the answer is
// one command rather than four fetches and a hand-built table — a hand-built
// table is where a number gets quietly copied into the wrong row.
//
// It does three things beyond printing numbers, and all three exist because
// of a mistake this project has already made:
//
//   · it prints the WIND-ONLY BAR from each bundle's own record, beside every
//     arm. A number without its baseline is not a result (ml/CLAUDE.md §3),
//     and the bar travels inside probe_kfold.json, so there is no excuse for
//     quoting one without the other.
//   · it refuses to call a gap significant. Overlapping confidence intervals
//     are not a test and non-overlapping ones are not either — the arms share
//     folds, months and most of their error, so the only honest comparison is
//     the paired one (scripts/paired_probe.py). The table says so in place of
//     a verdict.
//   · it names the arms that are BELOW the bar. A sweep can order U perfectly
//     and still be a null result about the head, and an ordering is the part
//     that looks like a finding.
//
//   node scripts/sweep_table.mjs --runs 127:U=1,128:U=2,129:U=4,130:U=8
//   node scripts/sweep_table.mjs --runs 127:U=1,128:U=2 --target move
import { readFileSync } from "node:fs";

const args = process.argv.slice(2);
const arg = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const REPO = process.env.GITHUB_REPOSITORY || "blauewelt/earth";
const TARGET = arg("--target", "rapid");
const SPEC = arg("--runs", "");
if (!SPEC) {
  console.error("usage: sweep_table.mjs --runs 127:U=1,128:U=2 [--target rapid]");
  process.exit(2);
}
const ARMS = SPEC.split(",").map((s) => {
  const [n, ...label] = s.split(":");
  return { run: Number(n), label: label.join(":") || `#${n}` };
});

const rows = [];
for (const a of ARMS) {
  const url = `https://raw.githubusercontent.com/${REPO}/ml-metrics/probes-${a.run}.json`;
  let bundle = null;
  try {
    const r = await fetch(url + "?cb=" + Date.now());
    if (r.ok) bundle = JSON.parse(await r.text());
  } catch { /* reported below */ }
  if (!bundle) { rows.push({ ...a, missing: "no probes-*.json on ml-metrics yet" }); continue; }
  const kf = bundle.files?.["probe_kfold.json"];
  // The bundle is keyed by the RUN DIRECTORY name ("actions"), not by target,
  // so take the first (and only) run key rather than hard-coding it — a local
  // re-score writes a different directory name and would otherwise read empty.
  const byRun = kf && kf[Object.keys(kf)[0]];
  const t = byRun?.[TARGET];
  if (!t) { rows.push({ ...a, missing: `bundle has no ${TARGET} k-fold` }); continue; }
  const head = bundle.files?.["probe_head.json"];
  const raw = bundle.files?.["probe_head_raw3x3.json"];
  const dip = bundle.files?.["dip_check.json"];
  rows.push({
    ...a,
    r: t.r_kfold_deseas, ci: t.ci95, n: t.n, rmse: t.rmse_sv,
    lp18: t.r_lowpass18,
    bar: t.wind_only_baseline?.r,
    head: head && (head.r ?? head.r_kfold ?? null),
    raw3: raw && (raw.r ?? raw.r_kfold ?? null),
    dip: dip && (dip.capture_pct ?? dip.capture ?? null),
    sha: bundle.files?.["provenance.json"]?.head_sha?.slice(0, 7),
  });
}

const f = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? "  —  " : Number(v).toFixed(d));
const pad = (s, w) => String(s).padEnd(w);
console.log(`\n${TARGET.toUpperCase()} · year-blocked k-fold (probe_kfold.py)\n`);
console.log(pad("arm", 10) + pad("run", 6) + pad("r", 8) + pad("95% CI", 18) +
            pad("rmse", 8) + pad("18-mo", 8) + "wind bar");
console.log("-".repeat(70));
for (const r of rows) {
  if (r.missing) { console.log(pad(r.label, 10) + pad("#" + r.run, 6) + r.missing); continue; }
  console.log(pad(r.label, 10) + pad("#" + r.run, 6) + pad(f(r.r), 8) +
              pad(`[${f(r.ci?.[0])}, ${f(r.ci?.[1])}]`, 18) +
              pad(f(r.rmse, 2), 8) + pad(f(r.lp18), 8) + f(r.bar));
}

const have = rows.filter((r) => !r.missing);
if (!have.length) {
  console.log("\nnothing to compare yet — the arms have not archived their probes.");
  process.exit(0);
}
const bar = have.find((r) => r.bar !== undefined && r.bar !== null)?.bar;
const under = have.filter((r) => bar != null && r.r <= bar);
const best = have.reduce((a, b) => (b.r > a.r ? b : a));
const worst = have.reduce((a, b) => (b.r < a.r ? b : a));

console.log(`\nspread ${f(worst.r)} (${worst.label}) → ${f(best.r)} (${best.label}) ` +
            `= ${f(best.r - worst.r)}`);
if (bar != null) {
  console.log(`wind-only bar on this tensor: ${f(bar)}`);
  if (under.length) {
    console.log(`BELOW THE BAR: ${under.map((r) => r.label).join(", ")} — an ordering of ` +
                `these arms is not a result about the head.`);
  }
}
console.log(
  `\nThis table ORDERS the arms; it does not test them. n=1 per arm, and the ` +
  `arms share\nfolds, months and most of their error, so neither overlapping ` +
  `nor separated CIs\nsettle anything. Before quoting a margin: ` +
  `scripts/paired_probe.py over the two\narms' per-month arrays, and a second ` +
  `seed (--seed-base 3).`);

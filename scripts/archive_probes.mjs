#!/usr/bin/env node
// Put a run's PROBE RESULTS somewhere permanent.
//
// Chris, 2026-08-10, on seeing that #119/#116/#115 archived two lines each:
// "is there something we should save/store from #116?"
//
// Yes, and the gap is structural rather than an oversight about one run. Three
// storage tiers exist and the scientific output sits in the wrong one:
//
//   ml-metrics branch   permanent   training CURVES (loss, lr, stage-2 steps)
//   GitHub releases     permanent   checkpoints, tensors, embeddings
//   Actions artifacts   30 DAYS     probe_kfold, probe_head, dip_check …
//
// The last row is the row with the answers in it. Every number this programme
// argues from — the k-fold correlations, the raw-3x3 control, the dip capture
// — lives only in an artifact that GitHub deletes after thirty days. The
// curves outlive the results.
//
// This writes them to ml-metrics as probes-<n>.json: additive, public, no
// credentials needed to read, and beside the curves they belong with.
//
//   node scripts/archive_probes.mjs --run-number 116 --dir ml/runs/f3_anchor_evalB \
//        [--token-file ~/.gh_pat]
import { readFileSync, readdirSync, existsSync } from "node:fs";
import { join } from "node:path";

const args = process.argv.slice(2);
const arg = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const RUN = arg("--run-number");
const DIR = arg("--dir");
const REPO = process.env.GITHUB_REPOSITORY || "blauewelt/earth";
const BRANCH = "ml-metrics";
if (!RUN || !DIR) {
  console.error("usage: archive_probes.mjs --run-number <n> --dir <run dir>");
  process.exit(2);
}
// ORDER MATTERS. An explicit --token-file wins, then the known PAT locations,
// and GITHUB_TOKEN only as the last resort. On a Vast box GITHUB_TOKEN is the
// job token and the files do not exist, so it is still reached. In this
// sandbox GITHUB_TOKEN is set to a PROXY credential beginning "prox" — reading
// it first sent that to GitHub and got a 401 that looked like a bad PAT.
const TOKEN = (() => {
  for (const p of [arg("--token-file"), "/home/claude/.gh_pat", `${process.env.HOME}/.gh_pat`]) {
    if (!p) continue;
    try {
      const t = readFileSync(p, "utf8").trim();
      if (t) return t;
    } catch { /* next */ }
  }
  return (process.env.GITHUB_TOKEN || "").trim();
})();
if (!TOKEN) { console.error("no token: cannot archive"); process.exit(1); }

// Everything a result is argued from, plus the provenance that says which
// tensor and code produced it. eval.json and metrics.jsonl are deliberately
// out — the curves are already on this branch.
const WANT = ["probe_kfold.json", "probe_head.json", "probe_head_raw3x3.json",
              "probe_head_raw.json", "probe_sequence.json", "dip_check.json",
              "rollout.json", "provenance.json"];
const bundle = { run_number: Number(RUN), archived_at: new Date().toISOString(), files: {} };
let n = 0;
for (const f of WANT) {
  const p = join(DIR, f);
  if (!existsSync(p)) continue;
  try { bundle.files[f] = JSON.parse(readFileSync(p, "utf8")); n++; }
  catch (e) { console.error(`  skipping ${f}: ${e.message}`); }
}
if (!n) {
  console.error(`no probe results in ${DIR} (looked for ${WANT.join(", ")})`);
  process.exit(1);
}
console.log(`bundling ${n} file(s) from ${DIR}: ${Object.keys(bundle.files).join(", ")}`);

// `token`, not `Bearer` — the rest of the repo's scripts use this form
// with this PAT, and Bearer returns 401 here.
const H = { Authorization: `token ${TOKEN}`, Accept: "application/vnd.github+json",
            "Content-Type": "application/json" };
const path = `probes-${RUN}.json`;
const url = `https://api.github.com/repos/${REPO}/contents/${path}`;
let sha;
const cur = await fetch(`${url}?ref=${BRANCH}`, { headers: H });
if (cur.ok) sha = (await cur.json()).sha;

const res = await fetch(url, {
  method: "PUT", headers: H,
  body: JSON.stringify({
    message: `archive probe results for run #${RUN}`,
    content: Buffer.from(JSON.stringify(bundle, null, 1)).toString("base64"),
    branch: BRANCH, ...(sha ? { sha } : {}),
  }),
});
if (!res.ok) {
  console.error("archive failed:", res.status, (await res.text()).slice(0, 300));
  process.exit(1);
}
// Assert the effect.
const raw = `https://raw.githubusercontent.com/${REPO}/${BRANCH}/${path}`;
for (let i = 0; i < 8; i++) {
  await new Promise((r) => setTimeout(r, 1500));
  if ((await fetch(raw + "?cb=" + i)).ok) { console.log(`archived: ${raw}`); process.exit(0); }
}
console.error("::warning::wrote it but it is not readable yet at " + raw);

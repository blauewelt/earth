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

// READ THROUGH THE API WHEN WE CAN. raw.githubusercontent serves a CDN copy
// that ignores cache-busting query strings and can lag minutes behind a
// write: a bundle re-archived with an extra file kept reading as the old
// three-file version here while the contents API already showed four. A
// table that is one edit stale is worse than one that fails, because it
// looks fine. The API is authoritative; raw is the unauthenticated fallback.
const TOKEN = (() => {
  for (const p of [arg("--token-file"), "/home/claude/.gh_pat", `${process.env.HOME}/.gh_pat`]) {
    if (!p) continue;
    try { const t = readFileSync(p, "utf8").trim(); if (t) return t; } catch { /* next */ }
  }
  return (process.env.GITHUB_TOKEN || "").trim();
})();

async function fetchBundle(run) {
  if (TOKEN) {
    try {
      const r = await fetch(
        `https://api.github.com/repos/${REPO}/contents/probes-${run}.json?ref=ml-metrics`,
        { headers: { Authorization: `Bearer ${TOKEN}`, Accept: "application/vnd.github+json" } });
      if (r.ok) {
        const j = await r.json();
        return JSON.parse(Buffer.from(j.content, "base64").toString("utf8"));
      }
      if (r.status === 404) return null;
    } catch { /* fall through to raw */ }
  }
  try {
    const r = await fetch(
      `https://raw.githubusercontent.com/${REPO}/ml-metrics/probes-${run}.json?cb=${Date.now()}`);
    if (r.ok) return JSON.parse(await r.text());
  } catch { /* reported below */ }
  return null;
}

const rows = [];
for (const a of ARMS) {
  const bundle = await fetchBundle(a.run);
  if (!bundle) { rows.push({ ...a, missing: "no probes-*.json on ml-metrics yet" }); continue; }
  const kf = bundle.files?.["probe_kfold.json"];
  // The bundle is keyed by the RUN DIRECTORY name ("actions"), not by target,
  // so take the first (and only) run key rather than hard-coding it — a local
  // re-score writes a different directory name and would otherwise read empty.
  const byRun = kf && kf[Object.keys(kf)[0]];
  // The codec probe is the CONTROL, so its absence must not blank the arm.
  // This bailed out here, which meant a bundle carrying the head k-fold but
  // no probe_kfold.json — the exact shape the archiver produced while it was
  // looking in the wrong directory — reported as "no results" with the
  // headline number sitting inside it.
  const t = byRun?.[TARGET] || {};
  if (!bundle.files?.["temporal.json"] && !byRun) {
    rows.push({ ...a, missing: "bundle has neither temporal.json nor probe_kfold.json" });
    continue;
  }
  // THE STAGE-2 NUMBER LIVES IN temporal.json, NOT HERE. probe_kfold pools
  // the FROZEN embeddings and never sees the temporal head, so for a sweep
  // over stage-2 choices its column is a CONTROL — it must be constant, and a
  // divergence would mean the codec was not actually held fixed. #116 (60k
  // head) and #125 (200k head) both read 0.631 [0.513, 0.732], which is how
  // this was noticed.
  const tj = bundle.files?.["temporal.json"];
  const prov = bundle.files?.["provenance.json"];
  const hk = tj?.rapid_probe_kfold;
  const z = tj?.["z_t+1"];
  rows.push({
    ...a,
    // headline: the head's own k-fold, when the run is new enough to have it
    r: hk?.r_kfold_deseas ?? null, ci: hk?.ci95 ?? null, n: hk?.n ?? null,
    split: tj?.rapid_probe?.r_deseasonalised ?? null,
    zratio: z && z.mse_persistence ? z.mse_model / z.mse_persistence : null,
    steps: tj?.steps ?? null,
    codec: t.r_kfold_deseas,                 // the control column
    // THE CONTROL IS THE TENSOR HASH, not the persistence baseline.
    //
    // The first version used z_mse_persistence, on the reasoning that it is
    // data-only and so must be bit-identical across runs sharing a codec and
    // a tensor. That is half right and it produced a FALSE ALARM on the first
    // clean experiment it saw: #140 (seed 0) and #141 (seed 1) are the same
    // box, the same tensor `adcbe700…` and the same codec, and their
    // baselines differ in the fourth digit — because the evaluation SAMPLE is
    // drawn with the run's seed. Data-only does not mean sample-independent.
    //
    // Measured: #121 and #140 — same box, same seed — agree to all sixteen
    // digits; #141 differs only in seed and moves. So persistence fingerprints
    // (tensor, seed) jointly, which is useless for a seed sweep.
    //
    // provenance.json now ships the tensor's sha256, which is exactly this
    // control and nothing else. Persistence stays as a fallback for runs
    // predating it, compared only WITHIN a seed.
    tensor: prov?.tensor?.sha256 ?? null,
    seed: tj?.seed ?? null,
    fingerprint: z?.mse_persistence ?? null,
    bar: t.wind_only_baseline?.r,
    dip: bundle.files?.["dip_check.json"]?.capture_pct ?? null,
  });
}

const f = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? "  —  " : Number(v).toFixed(d));
const pad = (s, w) => String(s).padEnd(w);
console.log(`\n${TARGET.toUpperCase()} · the HEAD's year-blocked k-fold ` +
            `(temporal.json → rapid_probe_kfold)\n`);
console.log(pad("arm", 12) + pad("run", 6) + pad("r", 8) + pad("95% CI", 18) +
            pad("36-mo", 8) + pad("z-ratio", 9) + "codec kfold");
console.log("-".repeat(76));
for (const r of rows) {
  if (r.missing) { console.log(pad(r.label, 12) + pad("#" + r.run, 6) + r.missing); continue; }
  console.log(pad(r.label, 12) + pad("#" + r.run, 6) + pad(f(r.r), 8) +
              pad(r.ci ? `[${f(r.ci[0])}, ${f(r.ci[1])}]` : "     —     ", 18) +
              pad(f(r.split), 8) + pad(f(r.zratio), 9) + f(r.codec));
}

const have = rows.filter((r) => !r.missing);
if (!have.length) {
  console.log("\nnothing to compare yet — the arms have not archived their probes.");
  process.exit(0);
}

// THE CONTROL, CHECKED RATHER THAN ASSUMED. Every arm freezing the same codec
// must report the same codec k-fold; if they do not, the arms differ in
// something other than the variable and nothing below is a comparison.
// Prefer the hash; fall back to persistence only among arms sharing a seed.
const tensors = [...new Set(have.map((r) => r.tensor).filter(Boolean))];
const seeds = [...new Set(have.map((r) => r.seed).filter((x) => x !== null))];
const sameSeed = seeds.length <= 1;
const fps = sameSeed
  ? [...new Set(have.map((r) => String(r.fingerprint)).filter((x) => x !== "null"))]
  : [];
const codecs = [...new Set(have.map((r) => f(r.codec)).filter((x) => x.trim() !== "—"))];
const noHash = have.filter((r) => !r.tensor).map((r) => "#" + r.run);
if (tensors.length > 1) {
  console.log(`\nCONTROL FAILED: the arms used DIFFERENT TENSORS ` +
              `(${tensors.map((t) => t.slice(0, 12)).join(" vs ")}). Boxes build ` +
              `their own copy and they have diverged; measured cost is 0.041 on ` +
              `the head k-fold. The ordering below is confounded.`);
} else if (tensors.length === 1 && !noHash.length) {
  console.log(`\ncontrol: every arm on tensor ${tensors[0].slice(0, 12)}… — one ` +
              `codec, one tensor, so the arms differ only in what was varied.`);
} else if (tensors.length === 1 && noHash.length) {
  // A HASH ON SOME ARMS IS NOT A CONTROL. The first version stopped at
  // "tensors.length === 1" and printed a clean pass for #131 vs #140 — which
  // are on demonstrably DIFFERENT tensors, because #131 predates provenance
  // and contributed no hash at all. A false pass is worse than the false
  // alarm it replaced: it certifies a confounded comparison.
  const fpAll = [...new Set(have.map((r) => String(r.fingerprint)))];
  if (sameSeed && fpAll.length > 1) {
    console.log(`\nCONTROL FAILED: ${noHash.join(", ")} carry no tensor hash ` +
                `(they predate provenance), and the persistence baselines ` +
                `disagree across arms at a single seed (${fpAll.join(" vs ")}). ` +
                `That means different tensors. The ordering below is confounded.`);
  } else {
    console.log(`\nCONTROL INCOMPLETE: ${noHash.join(", ")} carry no tensor hash, ` +
                `and the seeds differ, so persistence cannot substitute — it ` +
                `fingerprints (tensor, seed) jointly. The remaining arms are all ` +
                `on ${tensors[0].slice(0, 12)}…, but these cannot be confirmed.`);
  }
} else if (fps.length > 1) {
  console.log(`\nCONTROL FAILED: the persistence baseline differs across arms ` +
              `(${fps.join(" vs ")}). It is data-only and cannot depend on the ` +
              `model, so these runs did not share a codec or a tensor and the ` +
              `ordering below is confounded.`);
} else if (fps.length === 1) {
  console.log(`\ncontrol: persistence baseline ${fps[0]} on every arm — ` +
              `bit-identical, so the codec and the data path really were held ` +
              `fixed.` + (codecs.length === 1
                ? ` Codec k-fold ${codecs[0]} agrees.`
                : ` (The codec k-fold is absent from these bundles.)`));
} else if (codecs.length > 1) {
  console.log(`\nCONTROL FAILED: the codec k-fold differs across arms ` +
              `(${codecs.join(", ")}).`);
} else if (codecs.length === 1) {
  console.log(`\ncontrol: codec k-fold ${codecs[0]} on every arm, as it must ` +
              `be — that probe never sees the head.`);
} else {
  console.log("\nNO CONTROL AVAILABLE in these bundles — nothing verifies " +
              "that the arms held the codec fixed.");
}

const scored = have.filter((r) => r.r !== null);
if (!scored.length) {
  console.log("\nno arm carries rapid_probe_kfold — these ran before it existed " +
              "(added 2026-08-10). Their only stage-2 number is the 36-mo column.");
  process.exit(0);
}
const bar = have.find((r) => r.bar != null)?.bar;
const best = scored.reduce((a, b) => (b.r > a.r ? b : a));
const worst = scored.reduce((a, b) => (b.r < a.r ? b : a));
console.log(`spread ${f(worst.r)} (${worst.label}) → ${f(best.r)} (${best.label}) ` +
            `= ${f(best.r - worst.r)}`);
if (bar != null) {
  // THE BAR IS DIRECTLY COMPARABLE, and an earlier version of this line
  // hedged that it was not. Checked in the source: probe_kfold scores the
  // wind baseline with the SAME kfold_r, on the same deseasonalised RAPID
  // months, with the same year blocks and the same n=240 — only the features
  // differ (raw tau channels against the head's pooled hidden state). That is
  // exactly the comparison "does the model beat wind stress" means, so it is
  // a threshold and should be read as one.
  console.log(`wind-only bar, same protocol and same 240 months: ${f(bar)}`);
  const under = scored.filter((r) => r.r <= bar);
  if (under.length) {
    console.log(`BELOW THE BAR: ${under.map((r) => r.label).join(", ")} — these ` +
                `arms do not beat wind stress alone, so an ordering among them ` +
                `is not a result about the head.`);
  }
}
console.log(
  `\nThis table ORDERS the arms; it does not test them. n=1 per arm, and the ` +
  `arms share\nfolds, months and most of their error, so neither overlapping ` +
  `nor separated CIs\nsettle anything. Before quoting a margin: ` +
  `scripts/paired_probe.py over the two\narms' per-month arrays, and a second ` +
  `seed (--seed-base 3).`);

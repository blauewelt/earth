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
// A FOURTH, added 2026-08-21: it orders the UNPOOLED read-out and refuses to
// bar it with a pooled number. Chris: "we should not do pooled evals
// anywhere" / "move from pooled to unpooled when running evals". The
// mechanism is the argument — geostrophic transport at 26.5N is the
// east-minus-west contrast ACROSS the section, and every legacy_* column
// here is that section averaged away first (probe_kfold's Z.mean(1);
// temporal.py:2349's hid[:,-1].mean(0) for the stage-2 one). The pooled
// columns are KEPT and clearly labelled, because 183 archived bundles carry
// them and deleting them orphans the archive; they are never the verdict.
// When an unpooled head arrives with no unpooled bar, the "does it beat
// wind?" line is WITHHELD rather than answered against the pooled bar —
// switching one side of a comparison manufactures a result.
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
  // THE UNPOOLED READ-OUT, and its two UNPOOLED bars. ml/probe_head.py keeps
  // every (pixel, month) token and lets one query learn what to pool;
  // everything else on this row averages the section first. Since 2026-08-21
  // this is the verdict and the pooled columns are a labelled legacy
  // comparable (ml/CLAUDE.md §3). RAPID keeps the historical file names; a
  // non-default --target adds a suffix, which is why these are looked up by
  // the same rule ml/probe_head.py writes them under.
  const hsfx = TARGET === "rapid" ? "" : `_${TARGET}`;
  const hf = (kind) =>
    bundle.files?.[`probe_head${kind ? "_" + kind : ""}${hsfx}.json`] ?? null;
  const head = hf(""), head3 = hf("raw3x3"), headw = hf("wind");
  rows.push({
    ...a,
    // ---- THE VERDICT ---------------------------------------------------
    hr: head?.r_kfold_deseas ?? null,
    hci: head?.ci95 ?? null,
    hbar3: head3?.r_kfold_deseas ?? null,
    hbarw: headw?.r_kfold_deseas ?? null,
    // ---- LEGACY, POOLED, retained as the bridge to the archive ---------
    // `rapid_probe_kfold` is the STAGE-2 head's k-fold and it is pooled too:
    // ml/temporal.py:2349 reads `hid[:, -1].mean(0)  # pool along the
    // section`, and the in-training `stage2_probe` at :2181 does the same.
    // It was this table's headline until 2026-08-21. There is no unpooled
    // stage-2 read-out yet — that is the open follow-up, not something this
    // table can paper over — so the column stays, renamed to what it is.
    legacyPooledStage2: hk?.r_kfold_deseas ?? null,
    ci: hk?.ci95 ?? null, n: hk?.n ?? null,
    split: tj?.rapid_probe?.r_deseasonalised ?? null,
    zratio: z && z.mse_persistence ? z.mse_model / z.mse_persistence : null,
    steps: tj?.steps ?? null,
    legacyPooledCodec: t.r_kfold_deseas,     // the control column
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
    // POOLED: np.nanmean(tau, axis=1) over the same section
    // (ml/probe_kfold.py). Its own block now says `probe: "pooled-wind-only"`.
    legacyPooledBar: t.wind_only_baseline?.r,
    dip: bundle.files?.["dip_check.json"]?.capture_pct ?? null,
  });
}

const f = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? "  —  " : Number(v).toFixed(d));
const pad = (s, w) => String(s).padEnd(w);
console.log(`\n${TARGET.toUpperCase()} · UNPOOLED verdict (probe_head.json) ` +
            `with its own unpooled bars, then the LEGACY POOLED columns\n`);
console.log(pad("arm", 12) + pad("run", 6) + pad("UNPOOL r", 10) +
            pad("95% CI", 18) + pad("3x3 bar", 9) + pad("wind bar", 10) +
            pad("legacy_pooled_s2", 17) + pad("legacy_bar", 11) +
            "legacy_codec");
console.log("-".repeat(105));
for (const r of rows) {
  if (r.missing) { console.log(pad(r.label, 12) + pad("#" + r.run, 6) + r.missing); continue; }
  console.log(pad(r.label, 12) + pad("#" + r.run, 6) + pad(f(r.hr), 10) +
              pad(r.hci ? `[${f(r.hci[0])}, ${f(r.hci[1])}]` : "     —     ", 18) +
              pad(f(r.hbar3), 9) + pad(f(r.hbarw), 10) +
              pad(f(r.legacyPooledStage2), 17) +
              pad(f(r.legacyPooledBar), 11) + f(r.legacyPooledCodec));
}
console.log(
  `\nUNPOOLED IS THE VERDICT (ml/CLAUDE.md §3, 2026-08-21). Columns marked ` +
  `legacy_* read the\nsection MEAN — probe_kfold's Z.mean(1), and ` +
  `temporal.py:2349's hid[:,-1].mean(0) for the\nstage-2 one. Transport at ` +
  `26.5N is the east−west contrast ACROSS the section and that\nmean ` +
  `averages ~2.5 effective independent pixels of 265 (ml/project_amoc.py). ` +
  `They are\nkept because 183 archived bundles carry them; they are not a ` +
  `verdict.`);

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
const codecs = [...new Set(have.map((r) => f(r.legacyPooledCodec)).filter((x) => x.trim() !== "—"))];
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

// WHICH READ-OUT IS BEING ORDERED. The unpooled head when the bundles carry
// one; otherwise the legacy pooled stage-2 number, said out loud.
const unpooled = have.filter((r) => r.hr !== null);
const pooledOnly = have.filter((r) => r.legacyPooledStage2 !== null);
const useUnpooled = unpooled.length > 0;
const scored = useUnpooled ? unpooled : pooledOnly;
const key = (r) => (useUnpooled ? r.hr : r.legacyPooledStage2);
if (useUnpooled && unpooled.length < have.length) {
  const gap = have.filter((r) => r.hr === null).map((r) => "#" + r.run);
  console.log(`\nMIXED READ-OUTS: ${gap.join(", ")} carry no probe_head.json, ` +
              `so they have no unpooled\nnumber at all. They are NOT ordered ` +
              `below against the arms that do — a pooled number\nstanding in ` +
              `an unpooled column is the failure this whole change exists to ` +
              `remove.\nRe-dispatch them with head_probe: "true" (the ` +
              `workflow default since 2026-08-21).`);
}
if (!scored.length) {
  console.log("\nno arm carries probe_head.json OR rapid_probe_kfold — these " +
              "ran before either existed. Their only stage-2 number is the " +
              "36-mo column.");
  process.exit(0);
}
if (!useUnpooled) {
  console.log(`\nNO UNPOOLED NUMBER IN ANY BUNDLE. The ordering below is off ` +
              `the LEGACY POOLED\nstage-2 k-fold, which ml/CLAUDE.md §3 says ` +
              `is a comparability bridge and never a\nverdict. Read it as ` +
              `provenance for the archive, not as an answer; the arms need ` +
              `a\nre-dispatch with head_probe: "true".`);
}
const best = scored.reduce((a, b) => (key(b) > key(a) ? b : a));
const worst = scored.reduce((a, b) => (key(b) < key(a) ? b : a));
console.log(`\nspread ${f(key(worst))} (${worst.label}) → ${f(key(best))} ` +
            `(${best.label}) = ${f(key(best) - key(worst))}` +
            `${useUnpooled ? "  [unpooled]" : "  [LEGACY POOLED]"}`);

// THE BAR MUST BE THE SAME KIND OF READ-OUT AS THE NUMBER IT BARS.
//
// Until 2026-08-21 this block printed probe_kfold's wind_only_baseline
// beside whatever the headline was, on the reasoning that the protocol,
// months and year-blocks all match — which is true, and which is exactly
// what makes the mismatch invisible. The features do NOT match: that bar is
// np.nanmean(tau, axis=1), a section MEAN, and pairing it with an unpooled
// head credits the read-out's gain to the model. E-038's own numbers show
// the size of it: 0.660 pooled ridge → 0.691 unpooled head, against a pooled
// wind bar of 0.670. Which side of the bar the codec lands on is decided by
// the read-out, not by the codec.
//
// So the bar is chosen to MATCH, and when no matching bar exists the table
// refuses to draw one rather than reaching for the pooled number.
const unBar = have.find((r) => r.hbarw != null)?.hbarw;
const unBar3 = have.find((r) => r.hbar3 != null)?.hbar3;
const poBar = have.find((r) => r.legacyPooledBar != null)?.legacyPooledBar;
if (useUnpooled && unBar != null) {
  console.log(`unpooled wind bar (probe_head --raw --wind-only, same head, ` +
              `same folds): ${f(unBar)}` +
              (unBar3 != null ? ` · unpooled raw-3x3 bar: ${f(unBar3)}` : ""));
  const under = scored.filter((r) => key(r) <= unBar);
  if (under.length) {
    console.log(`BELOW THE BAR: ${under.map((r) => r.label).join(", ")} — these ` +
                `arms do not beat wind stress alone through the SAME unpooled ` +
                `read-out, so an ordering among them is not a result about the ` +
                `head.`);
  }
  if (poBar != null) {
    console.log(`(legacy pooled wind bar ${f(poBar)} is in the table for ` +
                `continuity with the archive. It is NOT this bar and the two ` +
                `are not interchangeable.)`);
  }
} else if (useUnpooled) {
  console.log(`NO MATCHED BAR: these bundles carry an unpooled head and NO ` +
              `unpooled wind bar` +
              (poBar != null ? `, only the pooled ${f(poBar)}` : "") +
              `.\nThe "does it beat wind?" verdict is WITHHELD rather than ` +
              `answered against a pooled\nbar — that comparison measures the ` +
              `read-out and reports it as the codec. Re-run\n` +
              `ml/probe_head.py --raw --wind-only (scripts/probes_run.sh does ` +
              `this automatically\nsince 2026-08-21).`);
} else if (poBar != null) {
  console.log(`legacy pooled wind bar, matched to the legacy pooled column ` +
              `above: ${f(poBar)}`);
  const under = scored.filter((r) => key(r) <= poBar);
  if (under.length) {
    console.log(`BELOW THE BAR (pooled vs pooled — matched, but neither is a ` +
                `verdict): ${under.map((r) => r.label).join(", ")}`);
  }
}
console.log(
  `\nThis table ORDERS the arms; it does not test them. n=1 per arm, and the ` +
  `arms share\nfolds, months and most of their error, so neither overlapping ` +
  `nor separated CIs\nsettle anything. Before quoting a margin: ` +
  `scripts/paired_probe.py over the two\narms' per-month arrays, and a second ` +
  `seed (--seed-base 3).`);

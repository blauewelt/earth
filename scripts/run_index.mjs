#!/usr/bin/env node
// Build ml/RUNS.md — the durable index of every ml-train run.
//
// Why this file exists: session reports quote runs as bare numbers ("#413")
// and the only link anyone could paste went to the Actions page, which shows
// a log and nothing about what the run WAS. status.html answers that, but
// only for the newest RUN_WINDOW runs; everything older simply has no
// readable address. This writes one, permanently: every run, one line of
// what it was for, its verdict, and the box it burned.
//
// Reads the PAT from a FILE, never argv — the permission classifier rightly
// blocks tokens on command lines (CLAUDE.md SS1), and scripts/git_api_push.mjs
// sets the convention this follows:
//
//   node scripts/run_index.mjs [--token-file ~/.gh_pat] [--repo blauewelt/earth]
//                              [--out ml/RUNS.md] [--no-boxes]
//
// One API call per 100 runs for the list, plus ONE PER RUN for the jobs
// endpoint (the runs API does not carry the runner name, and "which box" is
// the field that explains a 2.6 h run that should have taken 11). That is
// ~420 calls against a 5,000/h PAT budget. --no-boxes skips them.
import { readFileSync, writeFileSync } from "node:fs";

const arg = (name, dflt) => {
  const i = process.argv.indexOf(name);
  return i > 0 ? process.argv[i + 1] : dflt;
};
const REPO = arg("--repo", "blauewelt/earth");
const OUT = arg("--out", "ml/RUNS.md");
const WORKFLOW = arg("--workflow", "ml-train.yml");
const WANT_BOXES = !process.argv.includes("--no-boxes");
// Markdown links go to the phone reader, never to a GitHub blob (SS0b).
const DOCS = arg("--docs-base", "https://blauewelt.github.io/earth/docs.html");
const TOKEN = readFileSync(
  arg("--token-file", "/home/claude/.gh_pat").replace(/^~/, process.env.HOME),
  "utf8").trim();

async function api(path) {
  const res = await fetch("https://api.github.com" + path, {
    headers: {
      Authorization: `Bearer ${TOKEN}`,
      Accept: "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
    },
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${path}`);
  return res.json();
}

// ---- the doc, reduced to one line ------------------------------------------
// Two sources, curated first — exactly the order status.html's docOf() uses.
// ml/run_docs.json backfills the runs that predate the `doc` dispatch input
// (and can correct a bad run name); after 2026-08-08 the doc IS the run name,
// which the API returns as display_title.
let curated = {};
try {
  curated = JSON.parse(readFileSync("ml/run_docs.json", "utf8"));
} catch { /* no backfill file: display_title alone */ }

function docOf(run) {
  const c = curated[String(run.run_number)];
  if (c) return c;
  if (run.display_title && run.display_title !== "ml-train") return run.display_title;
  return null;
}

// First sentence or ~110 characters, whichever comes first. The docs are
// paragraphs now — #413's is 500 characters of dispatch rationale — and a
// table cell holding all of it is a table nobody reads.
const LIMIT = 110;
function summarise(text) {
  let s = String(text).replace(/\s+/g, " ").trim();
  if (!s) return null;
  // ". " ends a sentence; "0.25 degree" and "E-035a." do not. Require the
  // stop to be followed by a space and a capital or an opening quote, and to
  // sit after something longer than an abbreviation.
  const m = s.match(/^(.{20,}?[.!?])(\s+[A-Z(“"']|$)/);
  if (m && m[1].length <= LIMIT) return m[1];
  if (s.length <= LIMIT) return s;
  // cut on a word boundary, never mid-word
  const cut = s.slice(0, LIMIT);
  const sp = cut.lastIndexOf(" ");
  return (sp > 40 ? cut.slice(0, sp) : cut).replace(/[,;:.\s]+$/, "") + "…";
}

function expIdOf(run) {
  const hay = (docOf(run) || "") + " " + (run.display_title || "");
  const m = hay.match(/\bE-(\d{3})([a-z]?)\b/);
  return m ? "E-" + m[1] + (m[2] || "") : null;
}

// A markdown table cell cannot hold a raw pipe or a newline, and a stray
// backtick opens a code span that eats the rest of the row.
function cell(s) {
  return String(s).replace(/\s+/g, " ").replace(/\|/g, "\\|").replace(/`/g, "'").trim();
}

// ---- which box ---------------------------------------------------------------
// The self-hosted GPU boxes register under a per-instance label
// (gpu-box-45731106); GitHub's own runners report their image name. Both are
// worth knowing and they mean different things — #99 trained a 40M codec on a
// free 4-core CPU box purely because the dispatch left `runner` out.
function boxOf(jobs) {
  const names = new Set();
  for (const j of jobs || []) {
    const n = j.runner_name;
    const labels = (j.labels || []).join("+");
    if (n && /^gpu-box-/.test(n)) names.add("vast " + n.replace(/^gpu-box-/, ""));
    // GitHub's hosted runners are named per-job ("GitHub Actions 1000002305"),
    // which identifies nothing you can go and look at; the LABEL is the fact
    // that matters, because it is the image and therefore the hardware.
    else if (n && /^GitHub Actions\b/.test(n)) names.add(labels || "github-hosted");
    else if (n) names.add(n);
    // No runner_name at all: the job was never picked up (queued, then
    // cancelled, or the fleet was down). Say the labels it was WAITING for
    // in brackets rather than pretending a box ran it.
    else names.add("(" + (labels || "never assigned") + ")");
  }
  return names.size ? [...names].join(", ") : "—";
}

async function pool(items, n, fn) {
  const out = new Array(items.length);
  let i = 0;
  await Promise.all(Array.from({ length: Math.min(n, items.length) }, async () => {
    while (i < items.length) {
      const k = i++;
      out[k] = await fn(items[k], k);
    }
  }));
  return out;
}

// ---- fetch -------------------------------------------------------------------
const runs = [];
for (let page = 1; ; page++) {
  const j = await api(`/repos/${REPO}/actions/workflows/${WORKFLOW}/runs` +
                      `?per_page=100&page=${page}`);
  const got = j.workflow_runs || [];
  runs.push(...got);
  process.stderr.write(`\rlisted ${runs.length}/${j.total_count} runs`);
  if (!got.length || runs.length >= j.total_count) break;
}
process.stderr.write("\n");

// One row per run NUMBER: a re-run repeats the number under a new attempt,
// and the newest attempt is the one that says what happened.
const byNumber = new Map();
for (const r of runs) {
  const prev = byNumber.get(r.run_number);
  if (!prev || (r.run_attempt || 1) > (prev.run_attempt || 1)) byNumber.set(r.run_number, r);
}
const rows = [...byNumber.values()].sort((a, b) => b.run_number - a.run_number);

let boxes = new Map();
if (WANT_BOXES) {
  let done = 0;
  const got = await pool(rows, 6, async (r) => {
    let b = "—";
    try {
      b = boxOf((await api(`/repos/${REPO}/actions/runs/${r.id}/jobs?per_page=50`)).jobs);
    } catch { /* a run whose jobs were reaped reports no box, which is true */ }
    process.stderr.write(`\rboxes ${++done}/${rows.length}`);
    return [r.run_number, b];
  });
  boxes = new Map(got);
  process.stderr.write("\n");
}

// ---- write -------------------------------------------------------------------
function verdict(r) {
  if (r.status !== "completed") return r.status.replace(/_/g, " ");
  return r.conclusion || "completed";
}

const GROUP = 100;
const hi = rows.length ? rows[0].run_number : 0;
const lo = rows.length ? rows[rows.length - 1].run_number : 0;
const spanStart = rows.length ? rows[rows.length - 1].created_at.slice(0, 10) : "";
const spanEnd = rows.length ? rows[0].created_at.slice(0, 10) : "";

const L = [];
L.push("# Run index · ml-train");
L.push("");
L.push("**GENERATED FILE — do not hand-edit.** Rebuild it with " +
       "`node scripts/run_index.mjs --token-file ~/.gh_pat`, which reads the " +
       "GitHub Actions API and this repo's `ml/run_docs.json`. Any manual edit " +
       "is lost on the next rebuild.");
L.push("");
L.push(`${rows.length} runs, #${lo} → #${hi}, ${spanStart} → ${spanEnd}. ` +
       `Generated ${new Date().toISOString().slice(0, 16).replace("T", " ")} UTC.`);
L.push("");
L.push("Every row is addressable: `docs.html?f=ml/RUNS.md#run-413` opens this " +
       "file at run #413. A run still inside the status page's window is also " +
       "at `status.html#run-413`, which has its curves.");
L.push("");
L.push("The **summary** is the run's dispatch `doc` cut to its first sentence " +
       "(~110 characters). Runs before 2026-08-08 predate that input; theirs " +
       "come from `ml/run_docs.json`, and the few with neither say so. " +
       "**box** is the runner the job actually landed on — `vast <id>` is a " +
       "rented GPU box, `ubuntu-latest` is a free 4-core CPU runner, which is " +
       "the wrong hardware for a codec and has happened by omission.");
L.push("");

let head = null;
for (const r of rows) {
  const g = Math.floor((r.run_number - 1) / GROUP) * GROUP;
  if (g !== head) {
    if (head !== null) L.push("");
    head = g;
    L.push(`## Runs ${g + 1}–${g + GROUP}`);
    L.push("");
    L.push("| # | date | exp | summary | result | box |");
    L.push("|---|---|---|---|---|---|");
  }
  const doc = docOf(r);
  const summary = doc ? summarise(doc)
                      : (r.display_title && r.display_title !== "ml-train"
                          ? summarise(r.display_title) : null);
  const exp = expIdOf(r);
  L.push("| " + [
    // The anchor rides the run's own link, so the target is a real, visible
    // element — an empty <a id> in a table cell has no box to scroll to.
    `<a id="run-${r.run_number}" href="${r.html_url}">#${r.run_number}</a>`,
    r.created_at.slice(0, 10),
    exp ? `[${exp}](${DOCS}?f=ml/EXPERIMENTS.md#${exp.toLowerCase()})` : "—",
    summary ? cell(summary) : "*no doc recorded*",
    cell(verdict(r)),
    cell(boxes.get(r.run_number) || "—"),
  ].join(" | ") + " |");
}
L.push("");

writeFileSync(OUT, L.join("\n"));
console.log(`wrote ${OUT}: ${rows.length} rows, #${lo}–#${hi}`);

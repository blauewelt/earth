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
//                              [--out ml/RUNS.md] [--no-boxes] [--no-config]
//
// One API call per 100 runs for the list, plus ONE PER RUN for the jobs
// endpoint (the runs API does not carry the runner name, and "which box" is
// the field that explains a 2.6 h run that should have taken 11). That is
// ~420 calls against a 5,000/h PAT budget. --no-boxes skips them.
//
// The `config` column costs ONE more API call (the ml-metrics tree listing,
// so nothing is ever asked for that does not exist) and then reads
// raw.githubusercontent, which is NOT counted against the API budget at all.
// --no-config skips it.
import { readFileSync, writeFileSync } from "node:fs";

const arg = (name, dflt) => {
  const i = process.argv.indexOf(name);
  return i > 0 ? process.argv[i + 1] : dflt;
};
const REPO = arg("--repo", "blauewelt/earth");
const OUT = arg("--out", "ml/RUNS.md");
const WORKFLOW = arg("--workflow", "ml-train.yml");
const WANT_BOXES = !process.argv.includes("--no-boxes");
const WANT_CONFIG = !process.argv.includes("--no-config");
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

// ---- absolute, not relative -------------------------------------------------
// Standing rule, Chris 2026-08-19: *"[experiment descriptions must be]
// Absolute, not relative."* A doc that opens "RE-DISPATCH of #407, whose
// pinned box will not start…" tells a reader nothing at all until they have
// gone and read #407 — and by the time a run number reaches THIS file, the
// session that knew what it meant is long gone. The relative half is worth
// keeping (it is how the run came to exist); it is just never enough on its
// own.
//
// What this does NOT do is invent the missing half. Thirteen entries in
// ml/run_docs.json are honestly and entirely relative — #190's whole doc is
// "Cancelled: re-pinned as #195." and there is no absolute clause anywhere in
// it to carry. Those keep their relative line and nothing is appended: a
// fabricated purpose in a GENERATED index is worse than a short row, because
// nobody can tell it was generated.
const RELATIVE_OPEN = new RegExp(
  "^\\W*(?:re-?dispatch|re-?run|re-?try|re-?launch|resurrection|rescue|retry|" +
  "redo|repeat|continuation|attempt\\s+\\d|same\\s+(?:as|inputs)|replaces|" +
  "see\\s+#|as\\s+#|per\\s+#)\\b", "i");

// A summary is also relative when, with its run references struck out, almost
// nothing is left — "Cancelled: re-pinned as #195." reduces to nine words that
// name no job at all.
function isRelative(summary) {
  const s = String(summary || "");
  if (RELATIVE_OPEN.test(s)) return true;
  return /#\d+/.test(s) && s.replace(/#\d+/g, "").replace(/[^A-Za-z]+/g, " ").trim().length < 30;
}

// SENTENCES, not clauses. An earlier draft also split on em dashes, and that
// is how #114 got "— the documented Actions stall)." appended to a summary
// that already ended in those exact words: an em-dash split cuts a sentence in
// half and hands back the second half, which reads as a fragment and repeats
// text the reader has just been shown. The split requires the stop to be
// followed by a capital or an opening bracket, so "0.25 degree" and
// "18:11:25Z" survive intact.
function sentencesOf(text) {
  const s = String(text).replace(/\s+/g, " ").trim();
  const out = [];
  let start = 0;
  const re = /[.!?](?=\s+[A-Z(“"'#])/g;
  let m;
  while ((m = re.exec(s))) {
    out.push({ at: start, text: s.slice(start, m.index + 1).trim() });
    start = m.index + 1;
    while (s[start] === " ") start++;
    re.lastIndex = start;
  }
  if (start < s.length) out.push({ at: start, text: s.slice(start).trim() });
  return out;
}

// The first sentence, starting AFTER the summary already ends, that says
// something WITHOUT pointing at another run: no "#NNN" in it, and long enough
// to name a job rather than close a bracket. Returns null when the doc has
// none — which is the case this must get right, because the alternative is a
// generated file that invents a purpose nobody wrote.
//
// `after` is what stops the appended half from repeating the first: a sentence
// that begins before the summary's cut is already on the row.
function absoluteClause(text, after) {
  for (const s of sentencesOf(text)) {
    if (s.at < after) continue;
    const c = s.text.replace(/^[^A-Za-z0-9#]+/, "");
    if (/#\d+/.test(c)) continue;
    if (c.length < 25) continue;
    return c;
  }
  return null;
}

// ---- the structured config ---------------------------------------------------
// The machine half of the same rule: *"Structured, with fields: Num params,
// stage: encoder, data, … (you could even automatically render an experiment
// config)."* Same six fields as status.html's card strip, same fixed order,
// same single rule — PRINT WHAT THE RUN RECORDED, OMIT WHAT IT DID NOT. A
// field missing from this column means the run left no record of it; it does
// not mean zero and it does not mean the default.
//
//     params · stage · data · arch · steps×batch · resume
//
// Two sources on the ml-metrics branch, both read over raw.githubusercontent
// (uncounted against the API budget):
//   run-<n>.jsonl   the trainer's own `config` / `stage2_config` /
//                   `joint_config` / `resumed` records — params_M, shapes,
//                   steps, batch, tensor, C, T, resume. ~14 kB each.
//   probes-<n>.json files["provenance.json"].inputs — the verbatim dispatch
//                   block. The only place `window` lives, and therefore the
//                   only way to tell a headpub job from an eval-only one.
//                   These carry the full rollout payload and run to 6 MB, so
//                   they are fetched ONLY when the jsonl cannot answer.
const RAW = (path) => `https://raw.githubusercontent.com/${REPO}/ml-metrics/${path}`;

async function raw(path) {
  try {
    const r = await fetch(RAW(path));
    if (!r.ok) return null;
    return await r.text();
  } catch { return null; }
}

function recNum(v) {
  if (v === null || v === undefined || v === "" || v === false) return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}
function recPick(...vals) {
  for (const v of vals) if (v !== null && v !== undefined && v !== "" && v !== false) return v;
  return null;
}
const fmtInt = (n) => Number(n).toLocaleString("en-US");

// The records this column reads, pulled out of a run's metrics file. Deliberately
// a strict subset of status.html's parseJsonl(): the page charts these files,
// this only needs their headers.
function readMetrics(text) {
  const m = { cfg: null, s2: null, jc: null, resumed: null, loss: 0, sroll: false, s2steps: 0 };
  for (const line of String(text).split("\n")) {
    const t = line.trim();
    if (!t) continue;
    let o;
    try { o = JSON.parse(t); } catch { continue; }
    // A `config` record is a run's FIRST line, so anything before it belongs to
    // a previous run whose file was left in the workspace. Reset, exactly as
    // status.html does — one run's numbers must never be attributed to another.
    if (o.config) { Object.assign(m, { cfg: o.config, s2: null, jc: null, resumed: null, loss: 0, sroll: false, s2steps: 0 }); continue; }
    if (o.resumed) { m.resumed = o.resumed; continue; }
    if (o.joint_config) { m.jc = o.joint_config; continue; }
    if (o.stage2_config) { m.s2 = o.stage2_config; continue; }
    if (o.stage2_result && !m.s2) { m.s2 = o.stage2_result; continue; }
    if (o.sroll) { m.sroll = true; continue; }
    if (typeof o.stage2_step === "number") { m.s2steps++; continue; }
    if (typeof o.step === "number" && typeof o.loss_rec === "number") m.loss++;
  }
  return m;
}

// The stage, derived — the one field no config line carries. Identical rules to
// status.html's stageOf(), in the same order, and returning null by the same
// discipline: where the evidence cannot decide, the column shows the raw facts
// it DOES have (steps × batch, resume, both on the same row) rather than a
// label a reader would take for a record.
function stageOf(m, inp) {
  const w = inp && typeof inp.window === "string" ? inp.window : null;
  if (w && /^sroll:/.test(w)) return "sroll";
  if (w && /^headpub:/.test(w)) return "headpub";
  if (m && m.sroll) return "sroll";
  if (m && m.jc) return "joint";
  if (m && (m.s2 || m.s2steps)) return "stage-2";
  if (inp && recNum(inp.temporal_steps) > 0) return "stage-2";
  if (m && m.cfg && !m.loss && m.resumed &&
      recNum(m.resumed.at_step) !== null && recNum(m.cfg.steps) !== null &&
      recNum(m.resumed.at_step) >= recNum(m.cfg.steps)) {
    return w ? "eval-only" : null;   // headpub looks identical without `window`
  }
  if (m && m.loss) return "encoder";
  return null;
}

// The head's fields, and only when a head was actually trained.
// temporal_d_model/temporal_layers appear in EVERY dispatch block, including
// the ones with temporal_steps 0 where they describe a network that was never
// instantiated. Getting this wrong printed "head 0×0" on the status page for
// #413 before a 390px screenshot caught it.
function headField(s2, inp, s2key, inpKey) {
  if (s2 && s2[s2key] !== null && s2[s2key] !== undefined) return recNum(s2[s2key]);
  if (inp && recNum(inp.temporal_steps) > 0) return recNum(inp[inpKey]);
  return null;
}

function configOf(m, inp) {
  const f = [];
  const cfg = m && m.cfg, s2 = m && m.s2, jc = m && m.jc;

  // params — RECORDED ONLY. train.py and temporal.py count the real module's
  // parameters; re-deriving the number from d_model here would restate the
  // architecture in a second language and be wrong the first time a channel is
  // added (EXPERIMENTS.md records exactly that: 37,975,889 → 37,976,465).
  const pp = [];
  const cp = recPick(jc && jc.codec_params_M, cfg && cfg.params_M);
  const hp = recPick(jc && jc.temporal_params_M, s2 && s2.params_M);
  if (recNum(cp) !== null) pp.push(`${cp}M codec`);
  if (recNum(hp) !== null) pp.push(`${hp}M head`);
  if (pp.length) f.push(["params", pp.join(" + ")]);

  const st = stageOf(m, inp);
  if (st) f.push(["stage", st]);

  const data = recPick(inp && inp.tensor, cfg && cfg.data);
  if (data) {
    const shape = [];
    if (recNum(cfg && cfg.C) !== null) shape.push("C " + recNum(cfg.C));
    if (recNum(cfg && cfg.T) !== null) shape.push("T " + recNum(cfg.T));
    f.push(["data", String(data).replace(/\.npz$/, "") +
                    (shape.length ? ` (${shape.join(", ")})` : "")]);
  }

  const a = [];
  const dm = recNum(recPick(cfg && cfg.d_model, inp && inp.codec_d_model));
  const nl = recNum(recPick(cfg && cfg.n_layers, inp && inp.codec_layers));
  const nh = recNum(recPick(cfg && cfg.n_heads, inp && inp.codec_heads));
  const dd = recNum(recPick(cfg && cfg.d_dec, inp && inp.codec_d_dec));
  const dz = recNum(recPick(cfg && cfg.d_z, inp && inp.d_z));
  const pt = recNum(recPick(cfg && cfg.patch, inp && inp.patch));
  if (dm !== null && nl !== null) {
    a.push(`codec ${dm}×${nl}` + (nh !== null ? `, ${nh} heads` : "") +
           (dd !== null ? `, d_dec ${dd}` : ""));
  }
  if (dz !== null) a.push("d_z " + dz);
  if (pt !== null) a.push("patch " + pt);
  const hdm = headField(s2, inp, "d_model", "temporal_d_model");
  const hl = headField(s2, inp, "layers", "temporal_layers");
  if (hdm !== null && hl !== null) a.push(`head ${hdm}×${hl}`);
  if (a.length) f.push(["arch", a.join(", ")]);

  const sb = [];
  const steps = recNum(recPick(cfg && cfg.steps, inp && inp.steps));
  const batch = recNum(recPick(cfg && cfg.batch, inp && inp.batch));
  if (steps !== null) sb.push(fmtInt(steps) + (batch !== null ? ` × ${fmtInt(batch)}` : ""));
  const hs = headField(s2, inp, "steps", "temporal_steps");
  const hb = recNum(s2 && s2.batch);
  if (hs !== null) sb.push(`head ${fmtInt(hs)}` + (hb !== null ? ` × ${fmtInt(hb)}` : ""));
  if (sb.length) f.push(["steps×batch", sb.join(", ")]);

  // resume — verbatim, INCLUDING the leading "!". train.py reads that mark as
  // require_resume, so "!run-62" is a job that FAILS rather than quietly
  // training from scratch, and only the dispatch block still carries it
  // (train.py strips it before writing the config line).
  let res = null, have = false;
  if (inp && typeof inp.resume === "string") { res = inp.resume; have = true; }
  else if (cfg && Object.prototype.hasOwnProperty.call(cfg, "resume")) { res = cfg.resume; have = true; }
  // A recorded null IS a fact: train.py writes `"resume": a.resume or None`,
  // so null means "from scratch", not "unknown".
  if (have) f.push(["resume", res ? String(res) : "none"]);

  if (!f.length) return null;
  return f.map(([k, v]) => `${k} ${v}`).join(" · ");
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

// ---- the config column -------------------------------------------------------
// One API call lists the ml-metrics tree, so nothing is ever requested that
// does not exist (there are 413 runs and 220 metrics files; blind fetching
// would be ~600 requests, two thirds of them 404s). Everything after that is
// raw.githubusercontent, which costs nothing against the API budget, at the
// same concurrency 6 the boxes pass uses.
let configs = new Map();
if (WANT_CONFIG) {
  let have = new Set();
  try {
    const tree = await api(`/repos/${REPO}/git/trees/ml-metrics`);
    for (const t of tree.tree || []) have.add(t.path);
  } catch (e) {
    process.stderr.write(`\nml-metrics tree unreadable (${e.message}) — no config column\n`);
  }
  const wanted = rows.filter((r) => have.has(`run-${r.run_number}.jsonl`) ||
                                    have.has(`probes-${r.run_number}.json`));
  let done = 0, probesRead = 0;
  const got = await pool(wanted, 6, async (r) => {
    const n = r.run_number;
    let m = null;
    if (have.has(`run-${n}.jsonl`)) {
      const t = await raw(`run-${n}.jsonl`);
      if (t) m = readMetrics(t);
    }
    // The probes file is 6 MB at its worst because it embeds the whole rollout
    // payload, so it is fetched on exactly the two conditions status.html uses:
    // the metrics cannot say what stage the run was, or they carry no config
    // record at all (which is every job that trains nothing and therefore
    // writes no `config` line).
    let inp = null;
    const haveCfg = m && (m.cfg || m.s2 || m.jc);
    if (have.has(`probes-${n}.json`) && !(haveCfg && stageOf(m, null))) {
      const t = await raw(`probes-${n}.json`);
      if (t) {
        try {
          const j = JSON.parse(t);
          inp = (j.files && j.files["provenance.json"] && j.files["provenance.json"].inputs) || null;
          probesRead++;
        } catch { /* a truncated archive is not a config */ }
      }
    }
    process.stderr.write(`\rconfig ${++done}/${wanted.length}`);
    return [n, configOf(m, inp)];
  });
  configs = new Map(got.filter(([, v]) => v));
  process.stderr.write(`\nconfig: ${configs.size} rows, ${probesRead} provenance files read\n`);
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
L.push("The **config** column is RENDERED, not written: every field comes from " +
       "a record the run itself left on the `ml-metrics` branch — the " +
       "trainer's `config` / `stage2_config` line in `run-<n>.jsonl`, and the " +
       "verbatim dispatch block in `probes-<n>.json`. The fields are always in " +
       "the same order — **params · stage · data · arch · steps×batch · " +
       "resume** — and a field that is ABSENT means the run left no record of " +
       "it. Absent is not zero and not a default. `stage` is derived (see " +
       "`stageOf()` in the script) and is omitted rather than guessed where " +
       "the records cannot decide; the raw facts are in the same cell either " +
       "way. A `—` means the run archived nothing at all, which is what a " +
       "cancelled or never-started job leaves behind.");
L.push("");

let head = null;
for (const r of rows) {
  const g = Math.floor((r.run_number - 1) / GROUP) * GROUP;
  if (g !== head) {
    if (head !== null) L.push("");
    head = g;
    L.push(`## Runs ${g + 1}–${g + GROUP}`);
    L.push("");
    L.push("| # | date | exp | summary | config | result | box |");
    L.push("|---|---|---|---|---|---|---|");
  }
  const doc = docOf(r);
  let summary = doc ? summarise(doc)
                    : (r.display_title && r.display_title !== "ml-train"
                        ? summarise(r.display_title) : null);
  // Absolute, not relative. A summary that only points at another run keeps
  // its pointer AND carries the first clause of the doc that names the job on
  // its own terms — when the doc has one. When it does not, nothing is added.
  if (summary && doc && isRelative(summary)) {
    // Measured from the summary's own length, so the carried clause is always
    // text the row has not already shown.
    const abs = absoluteClause(doc, summary.replace(/…$/, "").length);
    if (abs) summary += " — " + summarise(abs);
  }
  const exp = expIdOf(r);
  L.push("| " + [
    // The anchor rides the run's own link, so the target is a real, visible
    // element — an empty <a id> in a table cell has no box to scroll to.
    `<a id="run-${r.run_number}" href="${r.html_url}">#${r.run_number}</a>`,
    r.created_at.slice(0, 10),
    exp ? `[${exp}](${DOCS}?f=ml/EXPERIMENTS.md#${exp.toLowerCase()})` : "—",
    summary ? cell(summary) : "*no doc recorded*",
    configs.has(r.run_number) ? cell(configs.get(r.run_number)) : "—",
    cell(verdict(r)),
    cell(boxes.get(r.run_number) || "—"),
  ].join(" | ") + " |");
}
L.push("");

writeFileSync(OUT, L.join("\n"));
console.log(`wrote ${OUT}: ${rows.length} rows, #${lo}–#${hi}`);

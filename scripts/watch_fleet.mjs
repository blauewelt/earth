#!/usr/bin/env node
// One compact progress line every N seconds, for the Monitor tool.
//
// Chris, 2026-08-10: "please add a monitor to report progress every 5 mins."
//
// Two rules it follows, both learned the hard way today. It reports what the
// BRANCHES say, never what a plan implies — every wrong status claim in this
// project came from describing intent instead of reading state. And it emits
// on every terminal outcome, not only on success: a watcher that greps for
// the happy path is silent through a crash, and silence looks exactly like
// "still running".
//
//   node scripts/watch_fleet.mjs [--every 300] [--runs 121,125,126,127]
import { readFileSync } from "node:fs";

const args = process.argv.slice(2);
const arg = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const EVERY = Number(arg("--every", 300)) * 1000;
const WATCH = arg("--runs", "").split(",").filter(Boolean).map(Number);
const REPO = process.env.GITHUB_REPOSITORY || "blauewelt/earth";
const TOKEN = (() => {
  for (const p of [arg("--token-file"), "/home/claude/.gh_pat", `${process.env.HOME}/.gh_pat`]) {
    if (!p) continue;
    try { const t = readFileSync(p, "utf8").trim(); if (t) return t; } catch { /* next */ }
  }
  return "";
})();
const H = TOKEN ? { Authorization: `Bearer ${TOKEN}`, Accept: "application/vnd.github+json" } : {};
const j = async (u) => { try { const r = await fetch(u, { headers: H }); return r.ok ? await r.json() : null; } catch { return null; } };
const txt = async (u) => { try { const r = await fetch(u); return r.ok ? await r.text() : null; } catch { return null; } };
const hhmm = (s) => `${Math.floor(s / 3600)}h${String(Math.round((s % 3600) / 60)).padStart(2, "0")}`;

const seen = new Map();          // run -> last reported state string
let lastAssets = -1;

async function tick() {
  // 30, not 12. The page has to be deep enough that a LONG-RUNNING job stays
  // in it as newer runs pile up: #121 (a 200,000-step CPU run, ~8 hours)
  // dropped out of a 12-run window the moment four arms were cancelled and
  // four re-dispatched, and simply stopped being reported — the watcher went
  // quiet about the one job that most needed watching, and quiet reads as
  // fine. A --runs entry cannot rescue what the fetch never returned.
  const runs = await j(`https://api.github.com/repos/${REPO}/actions/workflows/ml-train.yml/runs?per_page=30`);
  if (!runs) { console.log("· API unreachable this tick"); return; }
  const live = runs.workflow_runs.filter(
    (r) => WATCH.includes(r.run_number) || r.status !== "completed");

  const parts = [];
  for (const r of live) {
    const n = r.run_number;
    let note = r.status === "completed" ? r.conclusion : r.status;
    if (r.status === "in_progress") {
      const t = await txt(`https://raw.githubusercontent.com/${REPO}/ml-live-${n}/metrics.jsonl?cb=${Date.now()}`);
      const recs = (t || "").trim().split("\n").map((l) => { try { return JSON.parse(l); } catch { return null; } }).filter(Boolean);
      const emb = recs.filter((x) => x.embedding).pop();
      const s2 = recs.filter((x) => x.stage2_step ?? x.step).pop();
      // The EMBEDDING is reported separately from training: a run sitting at
      // "step 0" for ninety minutes is not stalled, it is embedding, and
      // conflating the two is what makes a healthy run look dead.
      if (s2 && (s2.stage2_step ?? s2.step)) {
        const st = s2.stage2_step ?? s2.step;
        // The TOTAL is not in the metrics record — it is in the published
        // plan, which is also the thing the curve is checked against. Reading
        // it from there rather than from the dispatch inputs means the ETA
        // and the graph can never disagree about how long the run is.
        const plan = await txt(`https://raw.githubusercontent.com/${REPO}/ml-metrics/plan-${n}.json?cb=${Date.now()}`);
        let tot = null;
        try { tot = Number(JSON.parse(plan).steps) || null; } catch { /* no plan yet */ }
        // ms/step from wall-clock over steps done, not a field: the trainer
        // logs elapsed seconds, and dividing is honest about restarts.
        const ms = s2.stage2_wall_s && st ? (s2.stage2_wall_s * 1000) / st : null;
        const rate = ms ? ` ${ms.toFixed(1)}ms/st` : "";
        const eta = tot && ms ? ` eta ${hhmm(((tot - st) * ms) / 1000)}` : "";
        note = `step ${st.toLocaleString()}${tot ? "/" + tot.toLocaleString() : ""}${rate}${eta}`;
      } else if (emb) {
        note = `embedding ${emb.embedding.pct}% (${emb.embedding.where})`;
      } else {
        note = "started, before stage 2";
      }
    }
    parts.push(`#${n} ${note}`);
    const key = `${r.status}/${r.conclusion}`;
    if (seen.has(n) && seen.get(n) !== key && r.status === "completed") {
      console.log(`#${n} FINISHED: ${r.conclusion} — https://github.com/${REPO}/actions/runs/${r.id}`);
    }
    seen.set(n, key);
  }

  // A QUEUED JOB AGAINST AN IDLE RUNNER IS STUCK, NOT SLOW. This has
  // happened four times: runs sitting for up to 22 minutes while the API
  // reported the runners online and idle, cured within 90 seconds by a
  // cancel and re-dispatch. It is invisible in a progress line — "queued"
  // looks the same either way — so it needs its own signal, and the signal
  // has to be the RUNNERS endpoint rather than the run's own status, which
  // lags.
  const queued = live.filter((r) => r.status === "queued").length;
  if (queued) {
    const rs = await j(`https://api.github.com/repos/${REPO}/actions/runners`);
    const idle = (rs?.runners || []).filter((r) => r.status === "online" && !r.busy);
    if (idle.length) {
      console.log(`STALL: ${queued} run(s) queued while ${idle.length} runner(s) ` +
                  `are online and idle (${idle.map((r) => r.name).join(", ")}). ` +
                  `Cancel and re-dispatch — that has cleared it every time. Do ` +
                  `NOT restart the boxes; they would lose their warm caches.`);
    }
  }

  // The embedding cache is the thing the E-009 staging waits on, so its
  // arrival is an event, not a field in a status line nobody re-reads.
  const rel = await j(`https://api.github.com/repos/${REPO}/releases/tags/embed-cache-v1`);
  const nAssets = rel ? rel.assets.length : -1;
  if (lastAssets >= 0 && nAssets > lastAssets) {
    console.log(`EMBED CACHE PUBLISHED: ${nAssets} chunk(s) on embed-cache-v1 — ` +
                `a box that has never seen this codec can now pull Z instead of ` +
                `spending 95 minutes rebuilding it`);
  }
  lastAssets = nAssets;

  console.log(`${new Date().toISOString().slice(11, 16)}Z · ` +
              (parts.join(" · ") || "no active runs") +
              ` · Z chunks ${nAssets < 0 ? "?" : nAssets}`);
}

for (;;) {
  await tick();
  await new Promise((r) => setTimeout(r, EVERY));
}

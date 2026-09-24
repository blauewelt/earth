// E-082: the family-1 FETCH-LANE KEEPER. Keeps at most MAX hosted fetch lanes
// (family1-build.yml, stage index,fetch) in flight from a queue that lives on
// the orphan branch `f1-queue` as lane_queue.json:
//
//   {"pending":[lane,...], "dispatched":[{...lane,"at":iso}], "done":[...], "failed":[...]}
//   lane = {store, stage, start, end, runner, extra_args, adapter_env [, retries]}
//
// It runs ON GITHUB (.github/workflows/family1-queue.yml): one hosted job loops
// for HOURS, and re-dispatches its own workflow before the budget ends until
// the queue is drained. A sandbox loop cannot do this -- it dies at the end of
// every turn, and the 16 slots sat idle for most of each hour.
//
//   node scripts/family1_queue.mjs             loop (MAX, HOURS, TICK_S from env)
//   node scripts/family1_queue.mjs --once      one tick, then exit (by hand)
//   node scripts/family1_queue.mjs --selftest  fixture tests of the pure logic
//
// Env: GH_TOKEN (required except --selftest), GITHUB_REPOSITORY, MAX (16),
// HOURS (5.5), TICK_S (180). No npm packages; Node 20's fetch.
//
// The logic is two PURE functions over data -- reconcile(queue, runs, nowIso)
// and toDispatch(queue, inflight, max) -- plus applySent(), so --selftest can
// exercise every transition without the network.

const REPO = process.env.GITHUB_REPOSITORY || "blauewelt/earth";
const TOKEN = process.env.GH_TOKEN || "";
const MAX = Number(process.env.MAX || 16);
const HOURS_IN = Number(process.env.HOURS || 5.5);
const TICK_S = Number(process.env.TICK_S || 180);
const BUILD_WF = "family1-build.yml";
const SELF_WF = "family1-queue.yml";
const BRANCH = "f1-queue";
const QFILE = "lane_queue.json";
// The job's timeout-minutes is 350 (5.83 h). A budget past ~5.6 h would let the
// job be killed before it re-dispatches itself, and then the chain just stops.
const HOURS_CAP = 5.6;

const LANE_KEYS = ["store", "stage", "start", "end", "runner", "extra_args", "adapter_env"];
const FAST_FAIL_MIN = 20;   // a failure younger than this is a connection-level (EDL) failure
const LOST_AFTER_MIN = 30;  // a dispatch with no run after this long was lost
const MATCH_SLACK_MIN = 2;  // a run's created_at may precede our `at` by clock skew
const RETRY_HOLD_MIN = 15;  // a fast-failed lane waits this long before it is re-dispatched (an Earthdata Login outage lasts minutes; three instant retries would burn all three inside one)
const MAX_RETRIES = 3;
const FETCH_STORES = ["irtb", "sst_acspo02", "swot", "xco2", "burned500"];

// ---------------------------------------------------------------- pure logic

const ms = (iso) => Date.parse(iso);
const laneOnly = (x) => Object.fromEntries(LANE_KEYS.map((k) => [k, x[k] ?? ""]));
const laneKey = (x) => LANE_KEYS.map((k) => x[k] ?? "").join("|");
// THE ADAPTER ENV IS PART OF THE NEEDLE (2026-09-24): sixteen gbif part
// lanes share store, stage and window and differ only in GBIF_PARTS, and a
// needle without it would pair each lane with whichever sibling's run was
// created first — a failure would then re-queue the wrong lane. The build's
// run-name carries adapter_env at its end for the same reason.
const titleNeedle = (l) => `family1 ${l.store} ${l.stage} ${l.start}..${l.end}`
  + (l.adapter_env ? ` ${l.adapter_env}` : "");
// The run-name joins optional fields with single spaces, so an empty
// probe_month or runner leaves two or three spaces in a row in the title.
const squash = (s) => String(s || "").replace(/\s+/g, " ").trim();

export function normalize(q) {
  return {
    ...q,
    pending: q.pending || [],
    dispatched: q.dispatched || [],
    done: q.done || [],
    failed: q.failed || [],
  };
}

// Returns {queue, events, unseen}: the new queue (the input is not mutated),
// what moved (for the log line), and how many dispatched lanes have no run
// listed yet -- those count as in flight, or a tick that follows a dispatch by
// less than GitHub's listing lag would overfill the slots.
export function reconcile(queue, runs, nowIso) {
  const q = normalize(structuredClone(queue));
  const now = ms(nowIso);
  const events = { done: [], retried: [], failed: [], lost: [], dropped: [] };
  const keep = [], toHead = [];
  const claimed = new Set();
  let unseen = 0;
  for (const d of q.dispatched) {
    const at = ms(d.at);
    const needle = titleNeedle(d);
    const seen = new Set(d.seen_runs || []);
    const run = runs
      .filter((r) => squash(r.display_title).includes(squash(needle))
        && ms(r.created_at) >= at - MATCH_SLACK_MIN * 60e3
        && !seen.has(r.id) && !claimed.has(r.id))
      .sort((a, b) => ms(a.created_at) - ms(b.created_at))[0];
    if (!run) {
      if (now - at > LOST_AFTER_MIN * 60e3) {
        const { at: _a, ...lane } = d;
        toHead.push(lane);
        events.lost.push(lane);
      } else { keep.push(d); unseen++; }
      continue;
    }
    if (run.id !== undefined) claimed.add(run.id);
    if (run.status !== "completed") { keep.push(d); continue; }
    if (run.conclusion === "success") {
      const e = { ...d, run_number: run.run_number, finished: run.updated_at };
      q.done.push(e);
      events.done.push(e);
      continue;
    }
    // Any other conclusion (failure, cancelled, timed_out, startup_failure...)
    // is a failure. Its length decides whether it was the fast connection-level
    // kind (the builder exits early when it cannot reach Earthdata Login).
    const t0 = ms(run.run_started_at || run.created_at);
    const mins = Math.round((ms(run.updated_at) - t0) / 60e3);
    if (mins < FAST_FAIL_MIN) {
      const retries = (d.retries || 0) + 1;
      const { at: _a, ...lane } = d;
      if (retries > MAX_RETRIES) {
        const e = { ...d, run_number: run.run_number, url: run.html_url,
          reason: "retries exhausted", finished: run.updated_at };
        q.failed.push(e);
        events.failed.push(e);
      } else {
        const back = { ...lane, retries, seen_runs: [...(d.seen_runs || []), run.id].filter((x) => x !== undefined),
          not_before: new Date(now + RETRY_HOLD_MIN * 60e3).toISOString() };
        toHead.push(back);
        events.retried.push({ ...back, run_number: run.run_number, mins, conclusion: run.conclusion });
      }
    } else {
      const e = { ...d, run_number: run.run_number, url: run.html_url,
        reason: `failed after ${mins} min — read the log`, finished: run.updated_at };
      q.failed.push(e);
      events.failed.push(e);
    }
  }
  q.dispatched = keep;
  q.pending = [...toHead, ...q.pending];
  // A LANE THAT IS ALREADY DONE IS NEVER DISPATCHED AGAIN. Measured 2026-09-23
  // on the seeded queue: three lanes re-queued by hand after a fast failure
  // had kept their old `dispatched` entry beside the new one, so the old entry
  // matched the old failed run and put a lane whose re-run had already parked
  // its parts back into pending. Dedupe by lane key against done AND within
  // pending itself (keep the first).
  const finished = new Set(q.done.map(laneKey));
  const seenKeys = new Set();
  q.pending = q.pending.filter((l) => {
    const k = laneKey(l);
    if (finished.has(k) || seenKeys.has(k)) { events.dropped.push(l); return false; }
    seenKeys.add(k);
    return true;
  });
  return { queue: q, events, unseen };
}

// The lanes to dispatch now, in queue order: as many as there are free slots.
export function toDispatch(queue, inflight, max, nowIso) {
  const free = Math.max(0, max - inflight);
  const now = nowIso ? ms(nowIso) : Date.now();
  // a held-back retry is skipped, not blocking: the lanes behind it go first
  return normalize(queue).pending.filter((l) => !l.not_before || ms(l.not_before) <= now).slice(0, free);
}

// Record dispatches that were actually sent: remove each from pending (first
// match by lane key) and append it to dispatched with its `at`. Used after a
// tick's dispatches, and again on a fresh copy after a stale-sha conflict.
export function applySent(queue, sent) {
  const q = normalize(structuredClone(queue));
  for (const s of sent) {
    const k = laneKey(s);
    if (q.dispatched.some((d) => laneKey(d) === k && d.at === s.at)) continue;
    const i = q.pending.findIndex((p) => laneKey(p) === k);
    if (i >= 0) q.pending.splice(i, 1);
    q.dispatched.push(s);
  }
  return q;
}

export function inflightRegex(queue) {
  const q = normalize(queue);
  const stores = new Set(FETCH_STORES);
  for (const l of [...q.pending, ...q.dispatched]) if (l.store) stores.add(l.store);
  const alt = [...stores].map((s) => s.replace(/[^A-Za-z0-9_]/g, "")).join("|");
  return new RegExp(`family1 (${alt}) index,fetch`);
}

// ---------------------------------------------------------------- GitHub I/O

class ApiError extends Error {
  constructor(msg, status, fatal) { super(msg); this.status = status; this.fatal = fatal; }
}

async function gh(method, path, body) {
  let r;
  try {
    r = await fetch(`https://api.github.com${path}`, {
      method,
      headers: {
        Authorization: `Bearer ${TOKEN}`,
        Accept: "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        ...(body ? { "Content-Type": "application/json" } : {}),
      },
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch (e) {
    throw new ApiError(`${method} ${path}: network: ${e.message}`, 0, false);
  }
  const text = await r.text();
  if (r.status === 401 || r.status === 403) {
    // A rate limit also answers 403; that one is transient.
    const limited = r.headers.get("x-ratelimit-remaining") === "0" || /rate limit/i.test(text);
    if (!limited) throw new ApiError(`${method} ${path}: ${r.status} ${text.slice(0, 300)}`, r.status, true);
  }
  let json = null;
  try { json = text ? JSON.parse(text) : null; } catch { /* not JSON */ }
  return { status: r.status, json, text };
}

function must(res, what, ok = [200]) {
  if (!ok.includes(res.status)) throw new ApiError(`${what}: ${res.status} ${res.text.slice(0, 300)}`, res.status, false);
  return res.json;
}

async function getQueue() {
  const j = must(await gh("GET", `/repos/${REPO}/contents/${QFILE}?ref=${BRANCH}`), `GET ${QFILE}@${BRANCH}`);
  const raw = Buffer.from(j.content, "base64").toString("utf8");
  return { queue: normalize(JSON.parse(raw)), sha: j.sha };
}

async function putQueue(queue, sha, message) {
  const content = Buffer.from(JSON.stringify(queue, null, 1) + "\n").toString("base64");
  return gh("PUT", `/repos/${REPO}/contents/${QFILE}`, { message, content, sha, branch: BRANCH });
}

async function listRuns(needFrom) {
  const base = `/repos/${REPO}/actions/workflows/${BUILD_WF}/runs?per_page=100`;
  const runs = [];
  for (const st of ["in_progress", "queued"]) {
    runs.push(...must(await gh("GET", `${base}&status=${st}`), `list ${st}`).workflow_runs);
  }
  // Completed: one page covers hours of lanes. Page further only when a
  // dispatched lane is older than everything the page reaches (a keeper that
  // resumes after a long gap), so a finished lane is never mistaken for lost.
  for (let page = 1; page <= 5; page++) {
    const got = must(await gh("GET", `${base}&status=completed&page=${page}`), "list completed").workflow_runs;
    runs.push(...got);
    const oldest = Math.min(...got.map((r) => ms(r.created_at)));
    if (got.length < 100 || !(needFrom < oldest)) break;
  }
  // A run that finished between two list calls shows up twice; the later
  // (completed) listing wins.
  return [...new Map(runs.map((r) => [r.id, r])).values()];
}

const sleep = (s) => new Promise((res) => setTimeout(res, s * 1000));

// Dispatches sent whose queue write has not landed yet; applied at the start
// of the next tick so a failed PUT can never cause a lane to be sent twice.
let unsaved = [];

async function tick() {
  const nowIso = new Date().toISOString();
  const { queue: q0, sha } = await getQueue();
  const carry = unsaved;
  const withCarry = applySent(q0, carry);
  const needFrom = Math.min(Infinity, ...withCarry.dispatched.map((d) => ms(d.at) - MATCH_SLACK_MIN * 60e3));
  const runs = await listRuns(needFrom);
  let { queue, events, unseen } = reconcile(withCarry, runs, nowIso);

  const rx = inflightRegex(queue);
  const active = runs.filter((r) => (r.status === "in_progress" || r.status === "queued") && rx.test(r.display_title || ""));
  const inflight = active.length + unseen;

  const sent = [];
  for (const lane of toDispatch(queue, inflight, MAX, nowIso)) {
    if (sent.length) await sleep(2.5);
    const at = new Date().toISOString();
    let res;
    try {
      res = await gh("POST", `/repos/${REPO}/actions/workflows/${BUILD_WF}/dispatches`,
        { ref: "main", inputs: laneOnly(lane) });
    } catch (e) {
      if (e.fatal) throw e;
      // Unknown whether it went through: record it as dispatched. If no run
      // appears, the lost-dispatch rule puts it back after 30 min; the other
      // choice risks a duplicate lane.
      console.log(`dispatch ${titleNeedle(lane)}: ${e.message} -- recorded, stopping this tick`);
      sent.push({ ...lane, not_before: undefined, at });
      break;
    }
    if (res.status !== 204) {
      console.log(`dispatch ${titleNeedle(lane)}: HTTP ${res.status} ${res.text.slice(0, 200)} -- left in pending, stopping this tick`);
      break;
    }
    sent.push({ ...lane, not_before: undefined, at });
  }
  unsaved = [...carry, ...sent];
  queue = applySent(queue, sent);

  const changed = JSON.stringify(queue) !== JSON.stringify(q0);
  if (changed) {
    const msg = `queue: +${sent.length} dispatched, ${events.done.length} done, ${events.failed.length} failed, ${events.retried.length + events.lost.length} requeued`;
    let res = await putQueue(queue, sha, msg);
    if (res.status === 409 || res.status === 422) {
      // Someone else wrote (a --once by hand). Redo the pure part on the fresh
      // copy with the same runs, then record our dispatches on it.
      const fresh = await getQueue();
      ({ queue, events } = reconcile(applySent(fresh.queue, carry), runs, nowIso));
      queue = applySent(queue, sent);
      res = await putQueue(queue, fresh.sha, msg);
    }
    must(res, `PUT ${QFILE}@${BRANCH}`, [200, 201]);
  }
  unsaved = [];

  console.log(`tick ${nowIso} inflight ${inflight} dispatched +${sent.length} pending ${queue.pending.length} ` +
    `done ${queue.done.length} failed ${queue.failed.length}` +
    (events.done.length ? ` (+${events.done.length} done)` : ""));
  for (const e of events.retried)
    console.log(`  retry ${e.retries}/${MAX_RETRIES}: ${titleNeedle(e)} (#${e.run_number} ${e.conclusion} after ${e.mins} min)`);
  for (const e of events.lost) console.log(`  lost dispatch, requeued: ${titleNeedle(e)}`);
  for (const e of events.dropped) console.log(`  dropped from pending (already done or duplicate): ${titleNeedle(e)}`);
  for (const e of events.failed) console.log(`  FAILED: ${titleNeedle(e)} #${e.run_number} ${e.reason} ${e.url || ""}`);
  return queue;
}

async function main() {
  if (!TOKEN) { console.error("GH_TOKEN is not set"); process.exit(1); }
  const once = process.argv.includes("--once");
  let hours = HOURS_IN;
  if (hours > HOURS_CAP) {
    console.log(`HOURS=${hours} exceeds ${HOURS_CAP} (the job's 350-min timeout must leave room to re-dispatch) -- using ${HOURS_CAP}`);
    hours = HOURS_CAP;
  }
  const t0 = Date.now();
  console.log(`family1 queue keeper: repo ${REPO} MAX ${MAX} HOURS ${hours} TICK_S ${TICK_S}${once ? " (--once)" : ""}`);
  for (;;) {
    let q = null;
    try {
      q = await tick();
    } catch (e) {
      if (e.fatal) { console.error(`FATAL: ${e.message}`); process.exit(1); }
      console.log(`tick ${new Date().toISOString()} skipped: ${e.message}`);
    }
    if (q && q.pending.length === 0 && q.dispatched.length === 0) {
      console.log("queue drained");
      process.exit(0);
    }
    if (once) process.exit(0);
    if ((Date.now() - t0) / 1000 + TICK_S >= hours * 3600) break;
    await sleep(TICK_S);
  }
  // Budget spent and the queue is not known to be empty: hand over to a fresh
  // run of this workflow. concurrency group family1-queue holds it until this
  // job ends.
  const res = await gh("POST", `/repos/${REPO}/actions/workflows/${SELF_WF}/dispatches`,
    { ref: "main", inputs: { max: String(MAX), hours: String(HOURS_IN) } });
  if (res.status === 204) {
    console.log(`budget of ${hours} h spent -- re-dispatched ${SELF_WF} (max ${MAX}, hours ${HOURS_IN})`);
    process.exit(0);
  }
  console.error(`budget spent -- re-dispatch of ${SELF_WF} FAILED: HTTP ${res.status} ${res.text.slice(0, 300)}`);
  process.exit(1);
}

// ---------------------------------------------------------------- self-test

function selftest() {
  const now = "2026-09-23T18:00:00Z";
  const ago = (min) => new Date(ms(now) - min * 60e3).toISOString();
  const lane = (store, start, end, extra = {}) => ({ store, stage: "index,fetch", start, end,
    runner: "ubuntu-latest", extra_args: "--push-parts", adapter_env: "", ...extra });
  const title = (l) => `family1 ${l.store} index,fetch ${l.start}..${l.end}  ${l.adapter_env || ""}`;
  const run = (id, l, status, conclusion, createdAgo, durMin) => ({
    id, run_number: 1000 + id, display_title: title(l), status, conclusion,
    created_at: ago(createdAgo), run_started_at: ago(createdAgo),
    updated_at: ago(createdAgo - (durMin ?? 0)), html_url: `https://x/${id}`,
  });

  const A = lane("sst_acspo02", "2010-01-01", "2010-06-30");  // succeeded
  const B = lane("irtb", "2012-01-01", "2012-12-31");         // failed at 9 min
  const C = lane("swot", "2023-01-01", "2023-06-30");         // failed at 45 min
  const D = lane("xco2", "2016-01-01", "2016-12-31", { retries: 3 }); // fast fail, retries 3
  const E = lane("burned500", "2005-01-01", "2005-12-31");    // not found, 40 min old
  const F = lane("burned500", "2006-01-01", "2006-12-31");    // not found, 5 min old
  const G = lane("irtb", "2013-01-01", "2013-12-31");         // running
  const P = lane("irtb", "2014-01-01", "2014-12-31");         // pending already

  const queue = {
    pending: [P],
    dispatched: [
      { ...A, at: ago(200) }, { ...B, at: ago(20) }, { ...C, at: ago(60) },
      { ...D, at: ago(15) }, { ...E, at: ago(40) }, { ...F, at: ago(5) }, { ...G, at: ago(90) },
    ],
    // done / failed deliberately absent: missing arrays are empty.
  };
  const runs = [
    run(1, A, "completed", "success", 199, 150),
    run(2, B, "completed", "failure", 19, 9),
    run(3, C, "completed", "failure", 59, 45),
    run(4, D, "completed", "cancelled", 14, 2),
    run(5, G, "in_progress", null, 89),
    run(6, E, "completed", "success", 400, 100),  // an OLD run of E's window: must not match
  ];
  const before = JSON.stringify(queue);
  const { queue: q, events, unseen } = reconcile(queue, runs, now);

  let fails = 0;
  const check = (name, cond) => { console.log(`${cond ? "ok  " : "FAIL"} ${name}`); if (!cond) fails++; };
  const has = (arr, l) => arr.find((x) => laneKey(x) === laneKey(l));

  check("input queue not mutated", JSON.stringify(queue) === before);
  check("A succeeded -> done with run_number, finished, at kept",
    !!has(q.done, A) && has(q.done, A).run_number === 1001 && has(q.done, A).finished === runs[0].updated_at && has(q.done, A).at === ago(200));
  check("B failed at 9 min -> pending (head block), retries 1, no `at`",
    q.pending.findIndex((x) => laneKey(x) === laneKey(B)) < q.pending.findIndex((x) => laneKey(x) === laneKey(P))
    && has(q.pending, B).retries === 1 && !("at" in has(q.pending, B)));
  check("B remembers its failed run (seen_runs [2])", JSON.stringify(has(q.pending, B).seen_runs) === "[2]");
  check("C failed at 45 min -> failed, 'failed after 45 min — read the log'",
    has(q.failed, C)?.reason === "failed after 45 min — read the log" && has(q.failed, C).run_number === 1003);
  check("D retries 3, fast cancel -> failed 'retries exhausted'", has(q.failed, D)?.reason === "retries exhausted");
  check("E not found, 40 min old -> head of pending, no retries bump",
    !!has(q.pending, E) && !has(q.dispatched, E) && has(q.pending, E).retries === undefined
    && q.pending.findIndex((x) => laneKey(x) === laneKey(E)) < q.pending.findIndex((x) => laneKey(x) === laneKey(P)));
  check("F not found, 5 min old -> stays dispatched", !!has(q.dispatched, F));
  check("G in progress -> stays dispatched", !!has(q.dispatched, G));
  check("P stays pending, last", laneKey(q.pending[q.pending.length - 1]) === laneKey(P));
  check("unseen = 1 (F)", unseen === 1);
  check("events: 1 done, 1 retried, 2 failed, 1 lost",
    events.done.length === 1 && events.retried.length === 1 && events.failed.length === 2 && events.lost.length === 1);
  check("counts: pending 3, dispatched 2, done 1, failed 2",
    q.pending.length === 3 && q.dispatched.length === 2 && q.done.length === 1 && q.failed.length === 2);

  // The re-dispatched B must not re-match its own fast-failed run #2, even
  // when the new `at` falls within the 2-min slack of that run's created_at.
  const q2 = { pending: [], dispatched: [{ ...has(q.pending, B), at: ago(18) }] };
  const r2 = reconcile(q2, runs, now);
  check("re-dispatched B ignores its seen run #2 -> still dispatched", r2.queue.dispatched.length === 1 && r2.queue.pending.length === 0);

  // toDispatch respects MAX.
  const many = { pending: Array.from({ length: 10 }, (_, i) => lane("irtb", `20${10 + i}-01-01`, `20${10 + i}-12-31`)) };
  check("toDispatch: 14 in flight, MAX 16 -> 2, in queue order",
    toDispatch(many, 14, 16).length === 2 && toDispatch(many, 14, 16)[0].start === "2010-01-01");
  check("toDispatch: 16 in flight -> 0", toDispatch(many, 16, 16).length === 0);
  check("toDispatch: 20 in flight -> 0", toDispatch(many, 20, 16).length === 0);
  check("toDispatch: 0 in flight, 10 pending -> 10", toDispatch(many, 0, 16).length === 10);
  {
    const held = { pending: [{ ...many.pending[0], not_before: "2026-09-23T12:20:00Z" }, many.pending[1]], dispatched: [], done: [], failed: [] };
    check("toDispatch: a held-back retry is skipped, the next lane goes", toDispatch(held, 15, 16, "2026-09-23T12:10:00Z")[0].start === many.pending[1].start);
    check("toDispatch: the hold expires", toDispatch(held, 15, 16, "2026-09-23T12:21:00Z")[0].start === many.pending[0].start);
    check("retry carries not_before 15 min after now", (() => { const b = has(q.pending, B); return b && b.not_before && Math.round((ms(b.not_before) - ms(now)) / 60e3) === 15; })());
  }

  // applySent moves the lanes and is idempotent on a re-apply.
  const sent = [{ ...many.pending[0], at: now }, { ...many.pending[1], at: now }];
  const q3 = applySent(many, sent);
  check("applySent: 2 moved to dispatched with at", q3.pending.length === 8 && q3.dispatched.length === 2 && q3.dispatched[0].at === now);
  check("applySent: re-apply is a no-op", JSON.stringify(applySent(q3, sent)) === JSON.stringify(q3));

  // Dispatch inputs carry only the workflow's inputs (retries/seen_runs would 422).
  check("laneOnly strips retries/seen_runs/at", JSON.stringify(Object.keys(laneOnly({ ...has(q.pending, B), at: now }))) === JSON.stringify(LANE_KEYS));

  // In-flight regex.
  const rx = inflightRegex({ pending: [] });
  check("inflight regex matches a fetch lane title", rx.test("family1 sst_acspo02 index,fetch 2010-01-01..2010-06-30 "));
  check("inflight regex ignores an assembly / probe", !rx.test("family1 irtb all 2012-01-01..2012-12-31 gpu-box-1") && !rx.test("family1 swot probe  2023-01"));

  // Sixteen gbif part lanes share the window; the adapter_env in the needle
  // is what pairs each lane with ITS run — here the second lane's run was
  // created first and failed, and it is the second lane that is retried.
  {
    const P1 = lane("gbif", "1600-01-01", "2026-12-31", { adapter_env: "GBIF_SNAPSHOT=2026-09-01 GBIF_PARTS=0:618" });
    const P2 = lane("gbif", "1600-01-01", "2026-12-31", { adapter_env: "GBIF_SNAPSHOT=2026-09-01 GBIF_PARTS=618:1236" });
    const qq = { pending: [], dispatched: [{ ...P1, at: ago(20) }, { ...P2, at: ago(20) }], done: [], failed: [] };
    const rr = [run(21, P2, "completed", "failure", 19, 5), run(22, P1, "in_progress", null, 18)];
    const { queue: q4, events: ev } = reconcile(qq, rr, now);
    check("adapter_env pairs a lane with its own run", ev.retried.length === 1 && ev.retried[0].run_number === 1021
      && !!has(q4.pending, P2) && !!has(q4.dispatched, P1) && !has(q4.pending, P1));
    check("inflight regex matches a part lane's title", inflightRegex({ pending: [P1] }).test(title(P1)));
  }

  console.log(fails ? `selftest: ${fails} FAILED` : "selftest: all passed");
  process.exit(fails ? 1 : 0);
}

if (process.argv.includes("--selftest")) selftest();
else main();

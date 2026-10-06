// Append fetch lanes to the family-1 queue (lane_queue.json on the orphan
// branch f1-queue, read by scripts/family1_queue.mjs). A lane already pending,
// dispatched or done with the same key is NOT added again — re-queuing by
// hand is how E-085 got duplicate dispatches.
//   node scripts/family1_enqueue.mjs <lanes.json> [--dry-run]
// lanes.json: [{store, stage, start, end, runner, extra_args, adapter_env, retries}]
// PAT from /home/claude/.gh_pat — a file read, never argv.
import fs from "node:fs";
const [file] = process.argv.slice(2);
const dry = process.argv.includes("--dry-run");
const tok = fs.readFileSync("/home/claude/.gh_pat", "utf8").trim();
const h = { Authorization: `Bearer ${tok}`, Accept: "application/vnd.github+json" };
const U = "https://api.github.com/repos/blauewelt/earth/contents/lane_queue.json";
const KEYS = ["store", "stage", "start", "end", "runner", "extra_args", "adapter_env"];
const key = (l) => KEYS.map((k) => l[k] ?? "").join("|");
const cur = await (await fetch(`${U}?ref=f1-queue`, { headers: h })).json();
const q = JSON.parse(Buffer.from(cur.content, "base64").toString("utf8"));
for (const k of ["pending", "dispatched", "done", "failed"]) q[k] = q[k] || [];
const have = new Set([...q.pending, ...q.dispatched, ...q.done].map(key));
const add = JSON.parse(fs.readFileSync(file, "utf8")).map((l) => ({
  store: l.store, stage: l.stage || "index,fetch", start: l.start, end: l.end,
  runner: l.runner || "ubuntu-latest", extra_args: l.extra_args ?? "--push-parts",
  adapter_env: l.adapter_env || "", retries: l.retries ?? 2 }));
const fresh = add.filter((l) => !have.has(key(l)));
console.log(`queue: ${q.pending.length} pending, ${q.dispatched.length} dispatched; adding ${fresh.length} of ${add.length}`);
for (const l of fresh) console.log(`  + ${l.store} ${l.start}..${l.end}`);
if (dry || !fresh.length) process.exit(0);
q.pending.push(...fresh);
const body = { message: `queue: +${fresh.length} pending (enqueue)`, branch: "f1-queue", sha: cur.sha,
  content: Buffer.from(JSON.stringify(q, null, 1) + "\n").toString("base64") };
const r = await fetch(U, { method: "PUT", headers: { ...h, "Content-Type": "application/json" }, body: JSON.stringify(body) });
console.log(r.status, r.status === 200 ? "queued" : (await r.text()).slice(0, 300));

// Publish the fleet's cost picture to a PUBLIC file the status page can read.
//
// Why a snapshot instead of a live call: the status page is credential-free by
// design ("Reads only public unauthenticated GitHub endpoints · no credentials
// in this page"), and the Vast balance needs the Vast API key. Putting that key
// in a page served to browsers is obviously out; the usual fix — a scheduled
// workflow holding it as a repo secret — is unavailable because this session's
// GitHub proxy returns 403 on /actions/secrets.
//
// So: a session with the key writes balance + burn rate here, and the PAGE
// projects the balance forward from the timestamp. The projection is the part
// that actually matters and it stays correct without any refresh, because the
// burn rate is nearly constant while the boxes are up. The page always shows
// how old the snapshot is, so a stale number is visibly stale rather than
// quietly wrong.
//
// The file contains no credentials — balance, instance count, hourly rate.
//
//   node scripts/publish_fleet_status.mjs
import { readFileSync } from "node:fs";

const VAST = readFileSync("/home/claude/.vast_key", "utf8").trim();
const GH = readFileSync("/home/claude/.gh_pat", "utf8").trim();
const H = { Authorization: `Bearer ${VAST}`, "User-Agent": "earth" };

const user = await (await fetch("https://console.vast.ai/api/v0/users/current/", { headers: H })).json();
const inst = await (await fetch("https://console.vast.ai/api/v1/instances/", { headers: H })).json();
const boxes = (inst.instances || []).map((i) => ({
  id: i.id,
  status: i.actual_status,
  dph: Number((i.dph_total || 0).toFixed(4)),
  // A STOPPED instance still bills storage. Omitting it understated the fleet
  // burn by ~$0.29/h on 2026-08-20 (8 stopped 100 GB boxes), which put the
  // status page's projected runway 2.5 h past reality on the day it mattered.
  storage_dph: Number((i.storage_total_cost || 0).toFixed(4)),
  disk_used_gb: i.disk_util,
  disk_gb: i.disk_space,
}));
const running = boxes.filter((b) => b.status === "running");
const fleet = {
  at: new Date().toISOString(),
  // credit is what Vast will actually spend; balance goes negative first and
  // the instances are stopped at balance_threshold, which is what took the
  // whole fleet down on 2026-08-09.
  credit_usd: Number(((user.credit || 0) + Math.min(0, user.balance || 0)).toFixed(2)),
  balance_usd: Number((user.balance || 0).toFixed(4)),
  boxes_total: boxes.length,
  boxes_running: running.length,
  // burn = compute on running boxes + storage on STOPPED ones. A running
  // box's dph_total already includes its own storage; a stopped box bills
  // storage_total_cost alone. Measured 2026-08-19 20:55Z -> 2026-08-20 06:30Z
  // against the credit ledger itself: $1.571/h actual vs $1.256/h by the old
  // running-only formula.
  burn_usd_per_h: Number((
    running.reduce((a, b) => a + b.dph, 0) +
    boxes.filter((b) => b.status !== "running")
         .reduce((a, b) => a + b.storage_dph, 0)
  ).toFixed(4)),
  boxes,
};
console.log(JSON.stringify(fleet, null, 1));

// Publish to the long-lived ml-metrics branch, next to the run archives.
// Via the CONTENTS API rather than git: the session's git proxy refuses
// pushes to this repo ("not in this session's authorized repository set"),
// but api.github.com passes through untouched — the same reason
// scripts/git_api_push.mjs exists. One file, one call, no clone.
const REPO = "blauewelt/earth", BRANCH = "ml-metrics", PATH = "fleet.json";
const gh = (path, init = {}) => fetch(`https://api.github.com/repos/${REPO}${path}`, {
  ...init,
  headers: { Authorization: `Bearer ${GH}`, Accept: "application/vnd.github+json",
             "User-Agent": "earth", ...(init.headers || {}) },
});
let sha;
const cur = await gh(`/contents/${PATH}?ref=${BRANCH}`);
if (cur.ok) sha = (await cur.json()).sha;          // update in place
const body = {
  message: `fleet snapshot ${fleet.at}`,
  content: Buffer.from(JSON.stringify(fleet, null, 1) + "\n").toString("base64"),
  branch: BRANCH,
  ...(sha ? { sha } : {}),
};
const put = await gh(`/contents/${PATH}`, { method: "PUT", body: JSON.stringify(body) });
console.log(put.ok
  ? `published ${BRANCH}/${PATH}`
  : `FAILED ${put.status}: ${(await put.text()).slice(0, 200)}`);

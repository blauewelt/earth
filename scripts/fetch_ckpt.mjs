#!/usr/bin/env node
// Pull a checkpoint out of the model-checkpoints-v1 release.
//
// Why this exists: on 2026-08-10 a 60,000-step stage-2 head spent nine hours
// living in exactly one place — one rented box's workspace — because the step
// that uploads it runs only after the whole probe ladder finishes. The
// release is the one storage in this project that outlives the container, the
// boxes, and any job timeout. Everything an eval needs should be fetchable
// from it by name, with no GPU and no Actions run.
//
//   node scripts/fetch_ckpt.mjs                       # list what is there
//   node scripts/fetch_ckpt.mjs f3_anchor41M__pixelmae.pt ml/runs/x/pixelmae.pt
//
// Public release, so the download needs no auth; the listing uses the PAT
// only because api.github.com is rate-limited unauthenticated.
import { createWriteStream, readFileSync } from "node:fs";
import { mkdir } from "node:fs/promises";
import { dirname } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";

const REPO = "blauewelt/earth", TAG = "model-checkpoints-v1";
let h = {};
try {
  h = { Authorization: `Bearer ${readFileSync("/home/claude/.gh_pat", "utf8").trim()}` };
} catch { /* unauthenticated is fine for a public repo, just rate-limited */ }
const rel = await (await fetch(
  `https://api.github.com/repos/${REPO}/releases/tags/${TAG}`,
  { headers: { ...h, Accept: "application/vnd.github+json" } })).json();
const assets = rel.assets || [];
const [name, dest] = process.argv.slice(2);
if (!name) {
  for (const a of assets) console.log(`${(a.size / 1e6).toFixed(1).padStart(7)} MB  ${a.name}`);
  process.exit(0);
}
const a = assets.find((x) => x.name === name);
if (!a) { console.error(`no asset "${name}" — run without arguments to list`); process.exit(1); }
const out = dest || name;
await mkdir(dirname(out), { recursive: true }).catch(() => {});
const r = await fetch(a.browser_download_url, { redirect: "follow" });
if (!r.ok) { console.error(`download failed ${r.status}`); process.exit(1); }
await pipeline(Readable.fromWeb(r.body), createWriteStream(out));
console.log(`${name} -> ${out} (${(a.size / 1e6).toFixed(1)} MB)`);

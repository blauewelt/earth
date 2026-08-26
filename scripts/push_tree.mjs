#!/usr/bin/env node
// Push a directory as ONE orphan commit force-updating a ref — the node-side
// pusher for tpu_status_mirror.py's --dump mode (the sandbox egress proxy
// rejects python POSTs to api.github.com; node passes — ml/CLAUDE.md §7).
// Orphan + force is the ml-live-* convention: a mailbox branch, not a ledger.
//   node scripts/push_tree.mjs --dir /tmp/dump --branch ml-live-tpu \
//        --message "tpu mirror" --token-file ~/.gh_pat
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";

const arg = (k, d) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : d; };
const DIR = arg("--dir"), BRANCH = arg("--branch"), MSG = arg("--message", "push_tree");
const REPO = arg("--repo", "blauewelt/earth");
const TOKEN = readFileSync(arg("--token-file"), "utf8").trim();
if (!DIR || !BRANCH) { console.error("need --dir and --branch"); process.exit(2); }

const H = { authorization: `Bearer ${TOKEN}`, accept: "application/vnd.github+json", "user-agent": "push-tree", "content-type": "application/json" };
const api = async (method, path, body) => {
  const r = await fetch(`https://api.github.com${path}`, { method, headers: H, body: body ? JSON.stringify(body) : undefined });
  if (!r.ok && !(method === "PATCH" && r.status === 422)) {
    throw new Error(`${method} ${path} -> ${r.status}: ${(await r.text()).slice(0, 300)}`);
  }
  return r.ok ? r.json() : null;
};

const walk = (d) => readdirSync(d).flatMap((n) => {
  const p = join(d, n);
  return statSync(p).isDirectory() ? walk(p) : [p];
});

const tree = [];
for (const p of walk(DIR)) {
  const { sha } = await api("POST", `/repos/${REPO}/git/blobs`,
    { content: readFileSync(p).toString("base64"), encoding: "base64" });
  tree.push({ path: relative(DIR, p), mode: "100644", type: "blob", sha });
}
const { sha: tsha } = await api("POST", `/repos/${REPO}/git/trees`, { tree });
const { sha: csha } = await api("POST", `/repos/${REPO}/git/commits`, { message: MSG, tree: tsha, parents: [] });
const patched = await api("PATCH", `/repos/${REPO}/git/refs/heads/${BRANCH}`, { sha: csha, force: true });
if (!patched) await api("POST", `/repos/${REPO}/git/refs`, { ref: `refs/heads/${BRANCH}`, sha: csha });
console.log(`pushed ${BRANCH} @ ${csha.slice(0, 9)} (${tree.length} files)`);

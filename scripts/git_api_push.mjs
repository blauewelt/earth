#!/usr/bin/env node
// Replay local commits to GitHub via the Git Data API — the documented
// fallback (CLAUDE.md §1) for when the sandbox git proxy refuses a push.
//
// Reads the PAT from a FILE (never argv/env-in-argv: the permission
// classifier rightly blocks tokens on command lines):
//
//   node scripts/git_api_push.mjs --token-file ~/.gh_pat \
//        [--repo blauewelt/earth] [--branch main] [--range origin/main..HEAD]
//
// Replays each commit in the range as a new API-side commit (same message,
// author and dates; NEW shas — run `git pull --rebase` afterwards, identical
// patches dedupe). Adding/updating .github/workflows/ files requires the
// token to carry the "Workflows" permission. Refuses non-fast-forward.
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";

const arg = (name, dflt) => {
  const i = process.argv.indexOf(name);
  return i > 0 ? process.argv[i + 1] : dflt;
};
const REPO = arg("--repo", "blauewelt/earth");
const BRANCH = arg("--branch", "main");
const RANGE = arg("--range", `origin/${BRANCH}..HEAD`);
const tokenFile = arg("--token-file", null);
if (!tokenFile) { console.error("--token-file is required"); process.exit(2); }
const TOKEN = readFileSync(tokenFile.replace(/^~/, process.env.HOME), "utf8").trim();

const git = (...args) => execFileSync("git", args, { encoding: "utf8" }).trim();
const gitB = (...args) => execFileSync("git", args);

async function api(method, path, body) {
  const res = await fetch(`https://api.github.com/repos/${REPO}${path}`, {
    method,
    headers: {
      Authorization: `Bearer ${TOKEN}`,
      Accept: "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      ...(body ? { "Content-Type": "application/json" } : {}),
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    throw new Error(`${method} ${path} -> ${res.status}: ${(await res.text()).slice(0, 300)}`);
  }
  return res.json();
}

const remoteHead = (await api("GET", `/git/ref/heads/${BRANCH}`)).object.sha;
const localBase = git("rev-parse", RANGE.split("..")[0]);
if (remoteHead !== localBase) {
  console.error(`remote ${BRANCH} is at ${remoteHead.slice(0, 9)}, local base is ` +
    `${localBase.slice(0, 9)} — rebase onto the current remote first (never force).`);
  process.exit(1);
}

const commits = git("rev-list", "--reverse", RANGE).split("\n").filter(Boolean);
console.log(`replaying ${commits.length} commit(s) onto ${REPO}@${BRANCH} (${remoteHead.slice(0, 9)})`);

let prevSha = remoteHead;
let prevTree = (await api("GET", `/git/commits/${prevSha}`)).tree.sha;
for (const c of commits) {
  const entries = [];
  for (const line of git("diff-tree", "--no-commit-id", "--name-status", "-r", c).split("\n").filter(Boolean)) {
    const [status, path] = line.split("\t");
    if (status === "D") { entries.push({ path, mode: "100644", type: "blob", sha: null }); continue; }
    const mode = git("ls-tree", c, "--", path).split(/\s+/)[0];
    const blob = gitB("show", `${c}:${path}`);
    let entry;
    try {
      const text = blob.toString("utf8");
      if (Buffer.from(text, "utf8").equals(blob)) entry = { path, mode, type: "blob", content: text };
    } catch { /* binary */ }
    if (!entry) {
      const { sha } = await api("POST", "/git/blobs",
        { content: blob.toString("base64"), encoding: "base64" });
      entry = { path, mode, type: "blob", sha };
    }
    entries.push(entry);
  }
  const tree = (await api("POST", "/git/trees", { base_tree: prevTree, tree: entries })).sha;
  const [an, ae, ad, ...msg] = git("log", "-1", "--format=%an%x00%ae%x00%aI%x00%B", c).split("\x00");
  const person = { name: an, email: ae, date: ad };
  const commit = await api("POST", "/git/commits", {
    message: msg.join("\x00"), tree, parents: [prevSha], author: person, committer: person,
  });
  console.log(`  ${c.slice(0, 9)} -> ${commit.sha.slice(0, 9)}  ${msg.join("").split("\n")[0].slice(0, 70)}`);
  prevSha = commit.sha; prevTree = commit.tree.sha;
}

const ref = await api("PATCH", `/git/refs/heads/${BRANCH}`, { sha: prevSha, force: false });
console.log(`${BRANCH} is now ${ref.object.sha}`);

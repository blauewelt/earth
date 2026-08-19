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
// Reproduces each commit in the range as the IDENTICAL object: same tree,
// parents, author, committer, message and signature, so GitHub returns the
// same sha and local history needs no reconciliation afterwards. (Until
// 2026-08-19 it paraphrased — author used as committer, message trimmed,
// signature dropped — which minted new shas and left every pushed commit
// Unverified.) Adding/updating .github/workflows/ files requires the token to
// carry the "Workflows" permission. Refuses non-fast-forward.
//
// It ALSO refuses to push a checkout of one branch onto another. --branch
// defaults to main, so a bare invocation from a feature branch silently
// replays that whole branch onto main — which is exactly how the unvalidated
// patch-codec architecture landed on main on 2026-08-07. The default is only
// safe when you are standing on the branch you are naming; --allow-cross-branch
// is the deliberate opt-out for the rare intentional case.
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

const OPTS = { maxBuffer: 256 * 1024 * 1024 };   // default 1 MB chokes on baked data files
const git = (...args) => execFileSync("git", args, { encoding: "utf8", ...OPTS }).trim();

// Guard: pushing HEAD to a branch you are not on is almost always a slip.
const HERE_BRANCH = git("rev-parse", "--abbrev-ref", "HEAD");
if (HERE_BRANCH !== BRANCH && !process.argv.includes("--allow-cross-branch")) {
  console.error(
    `refusing: you are on '${HERE_BRANCH}' but --branch is '${BRANCH}'.\n` +
    `  This would replay '${HERE_BRANCH}' onto '${BRANCH}'.\n` +
    `  Intended? add --allow-cross-branch. Meant the current branch? ` +
    `add --branch ${HERE_BRANCH}.`);
  process.exit(3);
}
const gitB = (...args) => execFileSync("git", args, OPTS);

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
// NOTHING TO PUSH is not a push. Previously this fell through, PATCHed the
// ref to the value it already had, and printed "main is now <sha>" — which
// reads exactly like a successful push and is how a no-op gets mistaken for
// work. It also put a needless write on a branch with other writers.
if (commits.length === 0) {
  console.log(`nothing to push: ${RANGE} is empty, ${BRANCH} already at ${remoteHead.slice(0, 9)}`);
  process.exit(0);
}
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
    // Inline small text; big files (baked data years, >1 MB) go through the
    // blob API first — one 160 MB tree request would be rejected, 46 blob
    // posts are not.
    if (blob.length <= 1024 * 1024) {
      try {
        const text = blob.toString("utf8");
        if (Buffer.from(text, "utf8").equals(blob)) entry = { path, mode, type: "blob", content: text };
      } catch { /* binary */ }
    }
    if (!entry) {
      const { sha } = await api("POST", "/git/blobs",
        { content: blob.toString("base64"), encoding: "base64" });
      entry = { path, mode, type: "blob", sha };
    }
    entries.push(entry);
  }
  const tree = (await api("POST", "/git/trees", { base_tree: prevTree, tree: entries })).sha;
  // REPRODUCE THE COMMIT OBJECT, DO NOT PARAPHRASE IT.
  //
  // A commit's sha is the hash of its bytes: tree, parents, author line,
  // committer line, optional gpgsig, blank line, message. Feed the API those
  // exact fields and it returns THE SAME SHA — the local and remote histories
  // are then literally the same objects and there is nothing to reconcile.
  //
  // This script used to paraphrase instead: it sent the author as the
  // committer too, and took the message through a .trim(). Both change the
  // bytes, so every push minted new shas, local looked permanently "ahead",
  // and the signature was dropped, which is why the pushed commits showed as
  // Unverified on GitHub. Measured 2026-08-19: passing the real committer,
  // the exact message and the signature reproduces the sha exactly.
  //
  // Source of truth is `git cat-file commit`, not `git log --format`, because
  // it is the object git actually hashed rather than a rendering of it.
  const rawObj = gitB("cat-file", "commit", c).toString("utf8");
  const sepAt = rawObj.indexOf("\n\n");
  const objHead = rawObj.slice(0, sepAt);
  const message = rawObj.slice(sepAt + 2);
  const localTreeSha = /^tree (\w+)/m.exec(objHead)[1];
  const localParents = [...objHead.matchAll(/^parent (\w+)/gm)].map((m) => m[1]);

  const person = (kind) => {
    const m = new RegExp(`^${kind} (.*) <(.*)> (\\d+) ([+-]\\d{4})$`, "m").exec(objHead);
    // Keep the ORIGINAL UTC offset. Git stores the offset in the object, so
    // the same instant written "+0200" and "Z" is different bytes and a
    // different sha.
    const off = m[4];
    const mins = (off[0] === "-" ? -1 : 1) *
                 (Number(off.slice(1, 3)) * 60 + Number(off.slice(3, 5)));
    const wall = new Date((Number(m[3]) + mins * 60) * 1000).toISOString().slice(0, 19);
    return { name: m[1], email: m[2], date: `${wall}${off.slice(0, 3)}:${off.slice(3)}` };
  };

  const sigMatch = /^gpgsig (.*(?:\n .*)*)$/m.exec(objHead);
  const signature = sigMatch
    ? sigMatch[1].split("\n").map((l) => (l.startsWith(" ") ? l.slice(1) : l)).join("\n")
    : null;

  const body = { message, tree, parents: [prevSha],
                 author: person("author"), committer: person("committer") };

  // A signature only means anything over the bytes it signed. If our
  // reconstruction changed the parent or the tree, the original signature is
  // no longer a signature OF this commit — passing it would produce an object
  // carrying a signature that cannot verify, which is worse than none. Attach
  // it only when the payload is genuinely identical.
  const samePayload = tree === localTreeSha &&
                      localParents.length === body.parents.length &&
                      localParents.every((p, i) => p === body.parents[i]);
  if (signature && samePayload) body.signature = signature;
  else if (signature) console.log(`  note: ${c.slice(0, 9)} loses its signature ` +
                                  `(tree or parent differs after replay)`);

  const commit = await api("POST", "/git/commits", body);
  const same = commit.sha === c;
  console.log(`  ${c.slice(0, 9)} -> ${commit.sha.slice(0, 9)}` +
              `${same ? " (identical)" : "         "}  ${message.split("\n")[0].slice(0, 60)}`);
  prevSha = commit.sha; prevTree = commit.tree.sha;
}

const ref = await api("PATCH", `/git/refs/heads/${BRANCH}`, { sha: prevSha, force: false });
console.log(`${BRANCH} is now ${ref.object.sha}`);

// FAST-FORWARD THE LOCAL BRANCH ONTO WHAT WE JUST CREATED.
//
// Every commit above is REPLAYED through the Git Data API, which mints a new
// sha for it (the printed `abc -> def` lines are exactly that). So the moment
// this script succeeds, the local branch points at the pre-push shas and the
// remote points at the new ones: identical trees, different history, and
// `git log origin/main..main` lists the commits as unpushed forever. On
// 2026-08-19 that cost three separate rounds of "you have unpushed commits" —
// each one investigated, each one a false alarm, and each one carrying a real
// risk of "fixing" it with a force-push that would clobber another writer.
//
// The fetch+reset is safe precisely because of the check that follows it: we
// only move local if its TREE already equals the remote's. If it does not,
// something else is going on (a concurrent commit, a partial push) and the
// right answer is to say so and leave the working state alone, not to reset
// over it.
try {
  git("fetch", "origin", BRANCH);
  const head = git("rev-parse", "HEAD");
  const lastReplayed = commits[commits.length - 1];
  const localTree = git("rev-parse", `HEAD^{tree}`);
  const pushedTree = git("rev-parse", `${ref.object.sha}^{tree}`);
  // THE GUARD IS IDENTITY, NOT RESEMBLANCE. Fast-forward only when HEAD is
  // precisely the last commit we replayed. Tree-equality alone is not enough:
  // an empty commit (or a message-only amend) has the same tree as its
  // parent, so a HEAD carrying one extra such commit would have looked
  // identical and been destroyed by the reset. Checking both means we move
  // local only when it is exactly the thing we just pushed, containing
  // exactly what we pushed.
  if (head !== lastReplayed) {
    console.log(`HEAD ${head.slice(0, 9)} is not the last replayed commit ` +
                `${lastReplayed.slice(0, 9)} — NOT fast-forwarding; local has moved ` +
                `since the push began`);
  } else if (localTree !== pushedTree) {
    console.log(`local tree ${localTree.slice(0, 9)} != pushed tree ${pushedTree.slice(0, 9)} — ` +
                `NOT fast-forwarding; the replay did not reproduce the tree`);
  } else if (git("status", "--porcelain", "--untracked-files=no")) {
    // Untracked files deliberately do NOT block: `reset --hard` leaves them
    // alone. Modified TRACKED files do, because it would discard them.
    console.log(`tracked files are modified — leaving local ${BRANCH} at its pre-push sha; ` +
                `run: git reset --hard ${ref.object.sha.slice(0, 9)} when clean`);
  } else if (head === ref.object.sha) {
    // The normal case now that commits reproduce exactly: local already IS
    // the pushed history, because they are the same objects.
    console.log(`local ${BRANCH} already at ${ref.object.sha.slice(0, 9)} — same objects, nothing to sync`);
  } else {
    git("reset", "--hard", ref.object.sha);
    console.log(`local ${BRANCH} fast-forwarded to ${ref.object.sha.slice(0, 9)} ` +
                `(replay produced different shas — check why: an unsigned commit, ` +
                `or a rewritten parent)`);
  }
} catch (e) {
  console.log(`could not sync local ${BRANCH}: ${e.message}`);
}

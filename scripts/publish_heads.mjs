#!/usr/bin/env node
// Publish stage-2 heads from run artifacts to the model-checkpoints release.
//
// E-011's eval needed six heads pulled from six run artifacts, VERIFIED
// against what each claimed to be, and uploaded as e010_*__temporal.pt —
// done by hand once, which is once more than a step this consequential
// should ever be done by hand: the verification is the part a tired session
// skips, and an eval over a mislabelled head produces a confident wrong
// table (the #10/#11 poisoned-cache lesson, one layer up).
//
// For each --runs entry the script: finds the run, downloads its probes-<n>
// artifact (temporal.pt rides in it), loads the checkpoint's OWN args and
// asserts (unroll, seed, steps, direct) match what the caller said it is,
// then uploads <prefix>_u{U}[_d{D}]_s{S}__temporal.pt to the release.
// Idempotent: an asset already present with the same byte size is skipped
// (DELETE+POST replacement opens a partial-asset window — publish_tensor.py
// learned that the hard way). Effect-checked by re-listing the release.
//
//   node scripts/publish_heads.mjs --runs 164:u1s0,171:u1s1 --prefix e012
//     [--steps 60000] [--token-file /home/claude/.gh_pat]
//
// The u/s in each entry is the CLAIM to verify, not the name to use — the
// name is always built from the checkpoint's own args.
import { execFileSync } from "node:child_process";
import { mkdtempSync, writeFileSync, readFileSync, existsSync, rmSync }
  from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const arg = (n, d) => { const i = process.argv.indexOf(n); return i > 0 ? process.argv[i + 1] : d; };
const REPO = arg("--repo", "blauewelt/earth");
const RUNS = (arg("--runs", "") || "").split(",").filter(Boolean);
const PREFIX = arg("--prefix", "");
const STEPS = Number(arg("--steps", "0"));
const TAG = arg("--tag", "model-checkpoints-v1");
const TOKEN = readFileSync(arg("--token-file", "/home/claude/.gh_pat"), "utf8").trim();
if (!RUNS.length || !PREFIX) {
  console.error("--runs 164:u1s0,... and --prefix are required"); process.exit(2);
}
const H = { Authorization: `Bearer ${TOKEN}`, Accept: "application/vnd.github+json" };

const api = async (path, init) => {
  const r = await fetch(`https://api.github.com${path}`, { headers: H, ...init });
  if (!r.ok) throw new Error(`${path} -> ${r.status}: ${(await r.text()).slice(0, 200)}`);
  return r.json();
};

// python reads the checkpoint — torch pickles are not a node format, and a
// paraphrased parser here would be exactly the mislabelling risk this script
// exists to remove.
const inspect = (pt) => JSON.parse(execFileSync("python3", ["-c", `
import json, sys, torch
tk = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
a = tk.get("args", {})
print(json.dumps({"unroll": a.get("unroll", 1), "seed": a.get("seed", 0),
                  "K": a.get("K"), "steps": a.get("steps"),
                  "direct": str(a.get("direct") or ""),
                  "stencil": a.get("stencil", 1),
                  "ring_km": float(a.get("ring_km", 0) or 0),
                  "step": tk.get("step")}))`, pt], { encoding: "utf8" }));

const rel = await api(`/repos/${REPO}/releases/tags/${TAG}`);
const assets = new Map(rel.assets.map((a) => [a.name, a]));
const runsList = await api(`/repos/${REPO}/actions/workflows/ml-train.yml/runs?per_page=50`);

let failed = 0;
const published = [];
for (const spec of RUNS) {
  const [numS, claim] = spec.split(":");
  const n = Number(numS);
  const m = /^u(\d+)s(\d+)$/.exec(claim || "");
  if (!m) { console.error(`${spec}: claim must look like u1s0`); failed++; continue; }
  const [cu, cs] = [Number(m[1]), Number(m[2])];
  const run = runsList.workflow_runs.find((r) => r.run_number === n);
  if (!run) { console.error(`#${n}: not in the last 50 runs`); failed++; continue; }
  const arts = await api(`/repos/${REPO}/actions/runs/${run.id}/artifacts`);
  const art = (arts.artifacts || []).find((a) => a.name === `probes-${n}`);
  if (!art || art.expired) {
    console.error(`#${n}: probes artifact missing/expired`); failed++; continue;
  }
  const dir = mkdtempSync(join(tmpdir(), `head-${n}-`));
  try {
    const zr = await fetch(
      `https://api.github.com/repos/${REPO}/actions/artifacts/${art.id}/zip`,
      { headers: H });
    if (!zr.ok) { console.error(`#${n}: artifact ${zr.status}`); failed++; continue; }
    const zip = join(dir, "a.zip");
    writeFileSync(zip, Buffer.from(await zr.arrayBuffer()));
    execFileSync("unzip", ["-o", "-q", zip, "-d", dir]);
    const pt = [join(dir, "temporal.pt"), join(dir, "actions", "temporal.pt")]
      .find((p) => existsSync(p));
    if (!pt) { console.error(`#${n}: no temporal.pt in artifact`); failed++; continue; }

    const info = inspect(pt);
    const bad = [];
    if (info.unroll !== cu) bad.push(`unroll ${info.unroll} != claimed ${cu}`);
    if (info.seed !== cs) bad.push(`seed ${info.seed} != claimed ${cs}`);
    if (STEPS && info.steps !== STEPS) bad.push(`steps ${info.steps} != ${STEPS}`);
    if (STEPS && info.step !== STEPS) bad.push(`trained to ${info.step}, not ${STEPS}`);
    // GEOMETRY vs PREFIX. The claim (u1s0) cannot tell an E-022 3x3 head from
    // an E-023 ring head: same unroll, same seed, same steps, same direct,
    // same byte size — the ONLY thing that distinguishes them is the name I
    // type. So the prefix's own markers are verified against the checkpoint:
    // `...s9...` must be stencil 9, `...s13...` stencil 13, `...r222...` a
    // 222 km ring, and a prefix claiming no ring must not carry one. Without
    // this, one mistyped --prefix publishes a ring head as its own control
    // and every number downstream is confidently wrong.
    // the LAST s<n> / r<n> group in the prefix, because every real prefix has
    // the marker after a digit ("e022s9") — a first attempt guarded the match
    // with [^0-9] to avoid catching the run number and thereby matched
    // NOTHING in every name actually in use, silently checking nothing at all
    const ps = [...PREFIX.matchAll(/s(\d+)/g)].pop();
    const pr = [...PREFIX.matchAll(/r(\d+)/g)].pop();
    if (ps && Number(ps[1]) !== info.stencil) {
      bad.push(`prefix says stencil ${ps[1]}, checkpoint is ${info.stencil}`);
    }
    if (pr && Math.round(info.ring_km) !== Number(pr[1])) {
      bad.push(`prefix says ring ${pr[1]} km, checkpoint is ${info.ring_km}`);
    }
    if (!pr && info.ring_km > 0) {
      bad.push(`checkpoint is a ${info.ring_km} km RING but the prefix `
               + `"${PREFIX}" does not say so`);
    }
    if (bad.length) {
      console.error(`#${n}: VERIFICATION FAILED — ${bad.join("; ")}`);
      failed++; continue;
    }
    const dpart = info.direct ? `_d${info.direct.replace(/,/g, "-")}` : "";
    const name = `${PREFIX}_u${info.unroll}${dpart}_s${info.seed}__temporal.pt`;
    const buf = readFileSync(pt);
    const have = assets.get(name);
    if (have && have.size === buf.length) {
      console.log(`#${n} -> ${name}: already published (${buf.length} bytes) — skip`);
      published.push(name); continue;
    }
    if (have) await api(`/repos/${REPO}/releases/assets/${have.id}`, { method: "DELETE" });
    const up = await fetch(
      `https://uploads.github.com/repos/${REPO}/releases/${rel.id}/assets?name=${name}`,
      { method: "POST",
        headers: { ...H, "Content-Type": "application/octet-stream",
                   "Content-Length": String(buf.length) },
        body: buf });
    if (!up.ok) {
      console.error(`#${n}: upload ${up.status}: ${(await up.text()).slice(0, 200)}`);
      failed++; continue;
    }
    console.log(`#${n} -> ${name}: published (${buf.length} bytes, `
      + `K=${info.K}, steps=${info.steps})`);
    published.push(name);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

// Assert the effect: re-list and confirm every published name is present.
const rel2 = await api(`/repos/${REPO}/releases/tags/${TAG}`);
const now = new Set(rel2.assets.map((a) => a.name));
const missing = published.filter((n) => !now.has(n));
if (missing.length) {
  console.error(`re-list is MISSING: ${missing.join(", ")}`); process.exit(1);
}
console.log(`${published.length}/${RUNS.length} heads on ${TAG}, re-list confirms.`);
process.exit(failed ? 1 : 0);

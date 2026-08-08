#!/usr/bin/env node
// Publish the raw source-data cache as a GitHub Release, and let cold boxes
// seed from it — "pre-fetching to GitHub" (user, 2026-08-08).
//
// WHY: a box with an empty /opt/earth-cache must download ~13 GB from the
// source archives, and the biggest (SIO's Argo server) 504s under load —
// four 10M-codec runs died that way in one day, and two boxes fetching in
// parallel plausibly CAUSED the 504s. GitHub Releases are a free CDN for
// public repos with a 2 GB per-asset cap, so the tars are split into
// 1900 MB chunks. The data is public (RG Argo: free with citation; NCEP
// R1: US public domain) — the release body carries the attribution.
//
// Usage:
//   node scripts/data_release.mjs publish     # tar+split+upload (sandbox)
//   node scripts/data_release.mjs list
// Seeding on a box is done by the workflow (seed-from-release step), not
// by this script.
import { readFileSync, createReadStream, statSync, readdirSync } from "node:fs";
import { execFileSync } from "node:child_process";

const TOKEN = readFileSync("/home/claude/.gh_pat", "utf8").trim();
const REPO = "blauewelt/earth";
const TAG = "data-cache-v1";
const h = { Authorization: `Bearer ${TOKEN}`, Accept: "application/vnd.github+json" };

async function api(method, url, body) {
  const r = await fetch(url, { method, headers: { ...h, ...(body ? { "Content-Type": "application/json" } : {}) },
                               body: body ? JSON.stringify(body) : undefined });
  if (!r.ok && r.status !== 422) throw new Error(`${method} ${url} -> ${r.status}: ${(await r.text()).slice(0, 200)}`);
  return r.json();
}

async function ensureRelease() {
  const r = await fetch(`https://api.github.com/repos/${REPO}/releases/tags/${TAG}`, { headers: h });
  if (r.ok) return r.json();
  return api("POST", `https://api.github.com/repos/${REPO}/releases`, {
    tag_name: TAG, name: "Source-data cache (RG Argo + NCEP wind)",
    body: "Raw source files the ml fetchers download, mirrored so training " +
          "boxes can cold-start without hitting the origin archives.\n\n" +
          "Contents: Roemmich-Gilson Argo climatology monthly NetCDF " +
          "(sio-argo.ucsd.edu; freely available, cite Roemmich & Gilson 2009) " +
          "and NCEP/NCAR R1 daily surface wind stress (downloads.psl.noaa.gov; " +
          "US public domain).\n\nAssets are `tar` archives split into <2 GB " +
          "chunks: `cat rg.tar.part* | tar x -C ml/cache` etc. Seeded " +
          "automatically by .github/workflows/ml-train.yml.",
    draft: false, prerelease: false,
  });
}

const [cmd] = process.argv.slice(2);
if (cmd === "publish") {
  const rel = await ensureRelease();
  console.log(`release ${rel.id} (${TAG})`);
  const have = new Set((rel.assets ?? []).map((a) => a.name));
  // STREAMED split: the sandbox has ~6 GB free against a 13 GB payload, so
  // tar is piped through `split --filter`, which hands each 1.9 GB chunk to
  // the uploader and deletes it before cutting the next. Peak disk: one
  // chunk. Serial by construction — which is also what the origin servers
  // needed from us all along.
  for (const d of ["rg", "wind_daily"]) {
    const probe = `${d}.tar.xaa`;
    if (have.has(probe)) { console.log(`skip ${d} (already uploaded)`); continue; }
    execFileSync("bash", ["-c",
      `cd ml/cache && tar cf - ${d} | split -b 1900m ` +
      `--filter='cat > /tmp/chunk.$FILE && node ${process.cwd()}/scripts/data_release.mjs ` +
      `upload-one /tmp/chunk.$FILE ${d}.tar.$FILE ${rel.id} && rm /tmp/chunk.$FILE' - ''`],
      { stdio: "inherit" });
  }
} else if (cmd === "upload-one") {
  const [path, name, relId] = process.argv.slice(3);
  const size = statSync(path).size;
  process.stdout.write(`  uploading ${name} (${(size / 1e9).toFixed(2)} GB) … `);
  const r = await fetch(
    `https://uploads.github.com/repos/${REPO}/releases/${relId}/assets?name=${name}`,
    { method: "POST",
      headers: { ...h, "Content-Type": "application/octet-stream",
                 "Content-Length": String(size) },
      body: createReadStream(path), duplex: "half" });
  console.log(r.status);
  if (!r.ok) { console.error((await r.text()).slice(0, 200)); process.exit(1); }
} else {
  const rel = await ensureRelease();
  for (const a of rel.assets ?? [])
    console.log(`${a.name}  ${(a.size / 1e9).toFixed(2)} GB  ${a.download_count} downloads`);
}

#!/usr/bin/env node
// Rent / start / stop / destroy the GPU box that runs our self-hosted Actions
// runner — driven from a session, so training capacity can follow the queue
// instead of a human clicking a console.
//
// WHY a marketplace box rather than GitHub's GPU larger runners: those are
// $0.052/min ($3.12/h) AND require an organisation on the Team plan, while a
// 4090 on Vast is ~$0.15-0.40/h with no org, no plan, and no 6-hour job cap.
// The 6-hour cap is the part that actually blocks us: the longer-history and
// larger-codec runs are days, not hours.
//
// CREDENTIALS: read from files, never argv or the command line — the
// permission classifier (correctly) blocks tokens in command lines, and a
// key in a shell history is a key on disk anyway. Note HOME is /root in this
// sandbox while the credential files live in /home/claude, so the paths are
// explicit rather than ~-relative (the PAT hit this exact trap first).
//   /home/claude/.vast_key   Vast.ai API key (Keys page, shown once)
//   /home/claude/.gh_pat     GitHub PAT. Measured 2026-08-07: it carries
//       Administration:write, so `create` MINTS the runner registration
//       token itself (POST /actions/runners/registration-token). That token
//       lives one hour — minting it at launch is strictly better than
//       pasting a long-lived one. A STOPPED instance keeps its registration
//       and reconnects on start, so stop/start is the cheap everyday loop
//       and create/destroy the rare one.
//
// Usage:
//   node scripts/gpu_box.mjs offers            # what a 4090 costs right now
//   node scripts/gpu_box.mjs create <offerId>  # rent + self-register runner
//   node scripts/gpu_box.mjs list              # instances + $/h + state
//   node scripts/gpu_box.mjs stop <id>         # keep the disk, stop the meter
//   node scripts/gpu_box.mjs start <id>
//   node scripts/gpu_box.mjs destroy <id>      # release everything
import { readFileSync } from "node:fs";

const arg = (n, d) => { const i = process.argv.indexOf(n); return i > 0 ? process.argv[i + 1] : d; };
const KEY = readFileSync(arg("--key-file", "/home/claude/.vast_key"), "utf8").trim();
const GH = arg("--token-file", "/home/claude/.gh_pat");
const REPO = arg("--repo", "blauewelt/earth");
const BASE = "https://console.vast.ai/api/v0";

/** One-hour runner registration token, minted at launch. */
async function registrationToken() {
  const r = await fetch(
    `https://api.github.com/repos/${REPO}/actions/runners/registration-token`,
    { method: "POST",
      headers: { Authorization: `Bearer ${readFileSync(GH, "utf8").trim()}`,
                 Accept: "application/vnd.github+json" } });
  if (!r.ok) throw new Error(`registration-token -> ${r.status}: ${(await r.text()).slice(0, 200)}`);
  return (await r.json()).token;
}

// Our jobs are CPU-heavy at the edges (tensor build, embedding passes) and
// GPU-bound in the middle, so the filter asks for real RAM and disk as well
// as the card — a 24GB 4090 next to 8GB of system RAM would OOM in
// build_dataset.py long before it ever reached torch.
const WANT = {
  gpu_name: { eq: "RTX 4090" },
  num_gpus: { eq: 1 },
  cpu_ram: { gte: 32000 },       // MB
  disk_space: { gte: 120 },      // GB
  reliability2: { gte: 0.98 },
  rentable: { eq: true },
  rented: { eq: false },
  type: "on-demand",             // NOT interruptible: preemption mid-run
  order: [["dph_total", "asc"]], // costs more in wasted hours than it saves
};

async function vast(method, path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method,
    headers: {
      Authorization: `Bearer ${KEY}`,
      Accept: "application/json",
      ...(body ? { "Content-Type": "application/json" } : {}),
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await res.text();
  if (!res.ok) throw new Error(`${method} ${path} -> ${res.status}: ${text.slice(0, 300)}`);
  return text ? JSON.parse(text) : {};
}

// The onstart script IS the provisioning: it installs the Actions runner,
// registers it against this repo with the (one-hour) registration token, and
// starts it. Everything else — python deps, the data cache — the workflow does.
//
// TWO container facts this has to respect, both of which bite the obvious
// version of this script:
//   · `svc.sh install` wants systemd. A Vast instance is a Docker container
//     and has none, so the runner is started with nohup instead.
//   · Vast RE-RUNS onstart on every container start, and the registration
//     token is only valid for an hour — so a stop/start a day later would
//     re-register with a dead token and abort. Hence the .runner check:
//     configure once, thereafter just run. That is also what makes
//     stop/start (rather than destroy/create) the cheap everyday loop.
function onstart(token) {
  return `#!/bin/bash
set -eux
export RUNNER_ALLOW_RUNASROOT=1
if [ ! -f /opt/runner/.runner ]; then
  apt-get update -qq && apt-get install -y -qq curl tar jq git
  mkdir -p /opt/runner && cd /opt/runner
  V=$(curl -sL https://api.github.com/repos/actions/runner/releases/latest | jq -r .tag_name | tr -d v)
  curl -sLo r.tar.gz "https://github.com/actions/runner/releases/download/v\${V}/actions-runner-linux-x64-\${V}.tar.gz"
  tar xzf r.tar.gz && rm r.tar.gz
  ./bin/installdependencies.sh
  ./config.sh --unattended --replace \\
    --url https://github.com/blauewelt/earth \\
    --token ${token} \\
    --name gpu-box --labels self-hosted,linux,x64,gpu,cuda \\
    --work /opt/runner/_work
fi
cd /opt/runner
nohup ./run.sh > /var/log/gh-runner.log 2>&1 &
nvidia-smi || true
`;
}

const [cmd, target] = process.argv.slice(2);   // `arg` is the flag reader above

if (cmd === "offers") {
  const r = await vast("POST", "/bundles/", WANT);
  for (const o of (r.offers || []).slice(0, 10))
    console.log(`${String(o.id).padEnd(10)} $${o.dph_total.toFixed(3)}/h  ` +
      `${o.gpu_name}  ${Math.round(o.cpu_ram / 1000)}GB ram  ` +
      `${Math.round(o.disk_space)}GB disk  rel ${(o.reliability2 * 100).toFixed(1)}%  ${o.geolocation ?? ""}`);
} else if (cmd === "create") {
  const token = await registrationToken();
  const r = await vast("PUT", `/asks/${target}/`, {
    client_id: "me",
    // CUDA + torch already in the image: the workflow's pip step then only
    // has to confirm it rather than pull 2.5GB of wheels on every run.
    image: "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime",
    disk: 120,
    onstart: onstart(token),
    runtype: "ssh",
    label: "earth-runner",
  });
  console.log(JSON.stringify(r, null, 2));
  console.log("\nrunner should appear within ~3 min at " +
    "https://github.com/blauewelt/earth/settings/actions/runners");
} else if (cmd === "list") {
  const r = await vast("GET", "/instances/");
  for (const i of r.instances || [])
    console.log(`${String(i.id).padEnd(10)} ${String(i.actual_status).padEnd(10)} ` +
      `$${(i.dph_total ?? 0).toFixed(3)}/h  ${i.gpu_name}  up ${Math.round((i.duration ?? 0) / 3600)}h  ${i.label ?? ""}`);
} else if (cmd === "stop" || cmd === "start") {
  console.log(JSON.stringify(await vast("PUT", `/instances/${target}/`, { state: cmd === "stop" ? "stopped" : "running" })));
} else if (cmd === "destroy") {
  console.log(JSON.stringify(await vast("DELETE", `/instances/${target}/`)));
} else {
  console.log("commands: offers | create <offerId> | list | stop <id> | start <id> | destroy <id>");
}

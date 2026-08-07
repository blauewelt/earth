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
// key in a shell history is a key on disk anyway.
//   ~/.vast_key   Vast.ai API key      (Keys page, shown once)
//   ~/.gh_runner_token   GitHub runner registration token (Settings ->
//                        Actions -> Runners -> New self-hosted runner; the
//                        ./config.sh --token value). Valid ONE HOUR, so it
//                        is only needed at first registration — a STOPPED
//                        instance keeps its registration and reconnects on
//                        start, which is why stop/start is the cheap loop
//                        and destroy/create is the rare one.
//
// Usage:
//   node scripts/gpu_box.mjs offers            # what a 4090 costs right now
//   node scripts/gpu_box.mjs create <offerId>  # rent + self-register runner
//   node scripts/gpu_box.mjs list              # instances + $/h + state
//   node scripts/gpu_box.mjs stop <id>         # keep the disk, stop the meter
//   node scripts/gpu_box.mjs start <id>
//   node scripts/gpu_box.mjs destroy <id>      # release everything
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const KEY = readFileSync(join(homedir(), ".vast_key"), "utf8").trim();
const BASE = "https://console.vast.ai/api/v0";

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
// starts it as a service so a reboot or a stop/start cycle comes back on its
// own. Everything else — python deps, the data cache — the workflow does.
function onstart(token) {
  return `#!/bin/bash
set -eux
export RUNNER_ALLOW_RUNASROOT=1
apt-get update -qq && apt-get install -y -qq curl tar jq git python3-pip
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
./svc.sh install && ./svc.sh start
nvidia-smi || true
`;
}

const [cmd, arg] = process.argv.slice(2);

if (cmd === "offers") {
  const r = await vast("POST", "/bundles/", WANT);
  for (const o of (r.offers || []).slice(0, 10))
    console.log(`${String(o.id).padEnd(10)} $${o.dph_total.toFixed(3)}/h  ` +
      `${o.gpu_name}  ${Math.round(o.cpu_ram / 1000)}GB ram  ` +
      `${Math.round(o.disk_space)}GB disk  rel ${(o.reliability2 * 100).toFixed(1)}%  ${o.geolocation ?? ""}`);
} else if (cmd === "create") {
  const token = readFileSync(join(homedir(), ".gh_runner_token"), "utf8").trim();
  const r = await vast("PUT", `/asks/${arg}/`, {
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
  console.log(JSON.stringify(await vast("PUT", `/instances/${arg}/`, { state: cmd === "stop" ? "stopped" : "running" })));
} else if (cmd === "destroy") {
  console.log(JSON.stringify(await vast("DELETE", `/instances/${arg}/`)));
} else {
  console.log("commands: offers | create <offerId> | list | stop <id> | start <id> | destroy <id>");
}

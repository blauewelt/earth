import { readFileSync } from "node:fs";
const KEY = readFileSync("/home/claude/.vast_key", "utf8").trim();
const GH = readFileSync("/home/claude/.gh_pat", "utf8").trim();

const v = await (await fetch("https://console.vast.ai/api/v1/instances/", {
  headers: { Authorization: `Bearer ${KEY}` } })).json();
for (const i of v.instances ?? []) {
  console.log(`instance ${i.id} ${i.actual_status} ${i.gpu_name}`);
  for (const k of ["dph_base", "dph_total", "storage_cost", "storage_total_cost",
                   "inet_up_cost", "inet_down_cost", "disk_space", "cpu_ram",
                   "cur_state", "status_msg", "machine_id", "geolocation"])
    if (i[k] !== undefined) console.log(`   ${k}: ${i[k]}`);
}
const r = await (await fetch(
  "https://api.github.com/repos/blauewelt/earth/actions/runners",
  { headers: { Authorization: `Bearer ${GH}`, Accept: "application/vnd.github+json" } })).json();
console.log(`\ngithub runners: ${r.total_count}`);
for (const x of r.runners ?? [])
  console.log(`   ${x.name} ${x.status} busy=${x.busy} labels=${x.labels.map(l => l.name).join(",")}`);

// Deregister a self-hosted runner by NAME (the Vast instance backing it is
// already destroyed; GitHub keeps the offline registration otherwise).
import { readFileSync } from "node:fs";
const GH = readFileSync("/home/claude/.gh_pat", "utf8").trim();
const h = { Authorization: `Bearer ${GH}`, Accept: "application/vnd.github+json" };
const base = "https://api.github.com/repos/blauewelt/earth/actions/runners";
const list = await (await fetch(base, { headers: h })).json();
for (const r of list.runners ?? []) {
  if (r.name !== process.argv[2]) continue;
  const d = await fetch(`${base}/${r.id}`, { method: "DELETE", headers: h });
  console.log(`delete ${r.name} (id ${r.id}) -> ${d.status}`);
}
const after = await (await fetch(base, { headers: h })).json();
console.log(`remaining runners: ${after.total_count}`);
for (const r of after.runners ?? []) console.log(`   ${r.name} ${r.status} busy=${r.busy}`);

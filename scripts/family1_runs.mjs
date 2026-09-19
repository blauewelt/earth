// List the family1-build workflow's runs (and, with --parts <family> <store>,
// the parked years on the Hub). PAT from /home/claude/.gh_pat, HF token from
// /home/claude/.hf_token — file reads only, never argv.
//   node scripts/family1_runs.mjs [--n 40] [--status in_progress|queued]
//   node scripts/family1_runs.mjs --parts family1_tf lst05
import fs from "node:fs";
const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
if (args[0] === "--parts") {
  const [fam, store] = [args[1], args[2]];
  const tok = fs.readFileSync("/home/claude/.hf_token", "utf8").trim();
  const h = { Authorization: `Bearer ${tok}` };
  const tree = async (p) => { const r = await fetch(`https://huggingface.co/api/datasets/chfrank/earth-tensors/tree/main/${p}?limit=100`, { headers: h }); return r.status === 200 ? r.json() : []; };
  const yrs = (await tree(`partials/${fam}/${store}`)).map(x => x.path.split("/").pop()).filter(y => /^\d{4}$/.test(y)).sort();
  const done = [], nodone = [];
  for (const y of yrs) ((await tree(`partials/${fam}/${store}/${y}`)).some(x => x.path.endsWith("done.json")) ? done : nodone).push(y);
  console.log(`${store}: ${yrs.length} year folders ${yrs[0] ?? "-"}..${yrs.at(-1) ?? "-"}; done ${done.length}; without done.json: ${nodone.join(",") || "none"}`);
  const s = await fetch(`https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/tensors/${fam}/${store}/store.json`);
  console.log(`published store.json: HTTP ${s.status}`);
} else {
  const tok = fs.readFileSync("/home/claude/.gh_pat", "utf8").trim();
  const h = { Authorization: `Bearer ${tok}`, Accept: "application/vnd.github+json" };
  const n = opt("--n", "40"), st = opt("--status", "");
  const r = await fetch(`https://api.github.com/repos/blauewelt/earth/actions/workflows/family1-build.yml/runs?per_page=${n}${st ? `&status=${st}` : ""}`, { headers: h });
  const j = await r.json();
  console.log(`total ${j.total_count}`);
  for (const w of j.workflow_runs ?? []) console.log(w.run_number, (w.conclusion ?? w.status).padEnd(11), w.updated_at.slice(0, 16), w.name.replace(/^family1 /, ""));
}

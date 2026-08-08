import { readFileSync } from "node:fs";
const TOKEN = readFileSync("/home/claude/.gh_pat", "utf8").trim();
const inputs = JSON.parse(process.argv[2]);
const ref = process.argv[3] || "main";
// Every run must say what it IS: `doc` becomes the run name (workflow
// run-name) and the description on status.html. Chris asked for this
// (2026-08-08) after a day of runs identifiable only by number.
if (!inputs.doc || !inputs.doc.trim()) {
  console.error("refusing to dispatch without a doc string — add " +
    '"doc": "one line: what this run is and why" to the inputs JSON');
  process.exit(2);
}
const res = await fetch(
  "https://api.github.com/repos/blauewelt/earth/actions/workflows/ml-train.yml/dispatches",
  {
    method: "POST",
    headers: {
      Authorization: `Bearer ${TOKEN}`,
      Accept: "application/vnd.github+json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ ref, inputs }),
  });
console.log(res.status, res.status === 204 ? "dispatched" : (await res.text()).slice(0, 300));

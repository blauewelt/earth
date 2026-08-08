import { readFileSync } from "node:fs";
const TOKEN = readFileSync("/home/claude/.gh_pat", "utf8").trim();
const inputs = JSON.parse(process.argv[2]);
const ref = process.argv[3] || "main";
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

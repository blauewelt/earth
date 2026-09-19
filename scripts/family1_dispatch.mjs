import fs from "node:fs";
const tok = fs.readFileSync("/home/claude/.gh_pat","utf8").trim();
const [wf, inputsJson] = process.argv.slice(2);
const r = await fetch(`https://api.github.com/repos/blauewelt/earth/actions/workflows/${wf}/dispatches`,{method:"POST",headers:{Authorization:`Bearer ${tok}`,"Accept":"application/vnd.github+json","Content-Type":"application/json"},body:JSON.stringify({ref:"main",inputs:JSON.parse(inputsJson)})});
console.log(r.status, r.status===204?"dispatched":await r.text());

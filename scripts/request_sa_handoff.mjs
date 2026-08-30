// Dispatch the repo's own `secret-handoff` workflow (see
// .github/workflows/secret-handoff.yml) to obtain the GCP service-account key
// this session needs to READ the TPU results bucket.
//
// Why this exists: the TPU trainers write their finished heads to a GCS
// bucket, and the roll evaluator reads heads from a GitHub release. Moving a
// head from one to the other is a sandbox-side step, and it needs read access
// to the bucket. The key lives only in Actions secrets; secrets cannot be read
// back through the API by design. The handoff workflow seals the secret to an
// ephemeral public key generated here, so the key material never appears in a
// log, a commit, an unencrypted artifact, or a chat transcript. The matching
// private key never leaves this sandbox.
//
// Usage:
//   node scripts/request_sa_handoff.mjs --pubkey-file /tmp/sh/pub.b64
//
// Then download the run's `sealed-secret` artifact and decrypt it per the
// header of the workflow file.
import fs from "node:fs";

const args = process.argv.slice(2);
const get = (flag, dflt) => {
  const i = args.indexOf(flag);
  return i >= 0 ? args[i + 1] : dflt;
};

const tokenFile = get("--token-file", "/home/claude/.gh_pat");
const pubkeyFile = get("--pubkey-file");
const secretName = get("--secret", "GCP_TPU_SA_KEY");
const repo = get("--repo", "blauewelt/earth");

if (!pubkeyFile) {
  console.error("need --pubkey-file (base64 of a PEM RSA public key)");
  process.exit(2);
}

const token = fs.readFileSync(tokenFile, "utf8").trim();
const pubkey = fs.readFileSync(pubkeyFile, "utf8").trim();

const headers = {
  Authorization: `Bearer ${token}`,
  "User-Agent": "earth",
  Accept: "application/vnd.github+json",
  "Content-Type": "application/json",
};

const res = await fetch(
  `https://api.github.com/repos/${repo}/actions/workflows/secret-handoff.yml/dispatches`,
  {
    method: "POST",
    headers,
    body: JSON.stringify({
      ref: "main",
      inputs: { secret_name: secretName, pubkey_b64: pubkey },
    }),
  },
);

if (res.status !== 204) {
  console.error(`dispatch failed: ${res.status} ${await res.text()}`);
  process.exit(1);
}
console.log(`dispatched secret-handoff for ${secretName}`);

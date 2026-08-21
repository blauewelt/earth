# Getting TPU access: the click-path, the quota names, and the arithmetic

**Status: OPERATOR CHECKLIST (2026-08-21). Nothing here is blocked on code —
the JAX port's eval gates are green (`ml/plans/JAX_PORT.md`). This is the
account setup that has to exist before the first TPU smoke run, written as
deep links because the console's own navigation is the slow part.**

Everything below is a link you can tap. The console remembers the last
project you had open, so **check the project selector in the blue bar on
every page** — the commonest failure here is editing the right setting in
the wrong project.

---

## 0 · Which account, and why it matters

Use a Google account **with no organization attached** — a plain consumer
account, not one that belongs to a managed Workspace or Cloud organization.

This is not a preference. Google enforces
`constraints/iam.disableServiceAccountKeyCreation` **by default on every
organization created on or after 2024-05-03**, and that constraint blocks
exactly the credential a session needs (step 5). A project with no
organization above it has no such policy to inherit, so key creation simply
works. If you start under an organization and hit "Service account key
creation is disabled", nothing in the project can fix it — only an
organization policy administrator can, and the cheaper move is a project
that was never under one.

A billing account must be attached, and TPUs are not in the free tier.

---

## 1 · Create the project

- [console.cloud.google.com/projectcreate](https://console.cloud.google.com/projectcreate)

Name it something you will recognise in a dropdown six months from now.
Note the **project ID** (not the display name) — every later step wants it.

---

## 2 · Enable the Cloud TPU API

- [Enable the Cloud TPU API](https://console.cloud.google.com/apis/library/tpu.googleapis.com)
- [Enable the Cloud Storage API](https://console.cloud.google.com/apis/library/storage.googleapis.com)

Quota does not appear on the quotas page until the API that owns it is
enabled, so do this **before** step 3 or the metric you are looking for will
not be in the list and it will look like your account is not allowed to have
it.

---

## 3 · Request the quota — and ask for v5e, not v6e

- [IAM & Admin → Quotas](https://console.cloud.google.com/iam-admin/quotas)

Filter by metric name. The names are not what the marketing calls them:

| what you want | the metric to filter for |
|---|---|
| spot v5e | `Preemptible TPU v5 lite pod cores per project per zone` |
| on-demand v5e | `TPU v5 lite pod cores per project per zone` |
| spot v6e | `Preemptible TPU v6e cores per project per zone` |
| on-demand v6e | `TPU v6e cores per project per zone` |

Tick the row for the zone you want, then **⋮ → Edit quota**, enter the new
value, submit.

**Ask for 8 cores of v5e.** The reason is auto-approval thresholds, which
Google publishes per metric: on-demand v5e auto-approves to **64 cores in
all zones**, and *preemptible* v5e auto-approves to **800–4,032 cores**
depending on zone. **Both v6e metrics auto-approve at 0 cores everywhere**,
so every single v6e request — however small — goes to a human reviewer and
waits. A v5e-8 is one host with 8 chips, which is the right first slice
anyway (see §6), and it is the request that comes back immediately.

Zones that serve v5e: `us-central1`, `us-east5`, `us-west1`, `us-west4`,
`us-south1`, `europe-west4`, `asia-southeast1`. Pick one and remember it —
the bucket in step 4 should be in the same region, or every epoch pays
cross-region egress on the tensor.

If the quota page shows the metric but the edit is refused, the message
names the reason; send it over rather than working around it.

---

## 4 · Create the staging bucket

- [Create a bucket](https://console.cloud.google.com/storage/create-bucket)

**Single region, same region as the TPU zone.** Standard storage class. The
tensors are the reason it exists: `family3_na025` is 2.98 GB compressed and
the pentad tensor is far larger, and they get pulled onto the TPU VM's local
disk at job start, exactly the way the fleet seeds from `data-cache-v1`
today.

---

## 5 · Create the service account and its key

- [IAM & Admin → Service accounts](https://console.cloud.google.com/iam-admin/serviceaccounts)

**Create service account**, then grant it these roles — narrow ones, not
Owner:

| role | why |
|---|---|
| TPU Admin | create, list and delete TPU VMs |
| Storage Object Admin | read and write the staging bucket's objects |
| Service Account User | lets it act as itself when a TPU VM is created |

Then open the account → **Keys** tab → **Add key** → **Create new key** →
**JSON** → Create. The file downloads once and **cannot be retrieved again**
— if it is lost, delete that key and make another.

Paste the JSON contents into chat. It goes into a claude.ai project doc on
the same footing as the CMEMS credentials (root `CLAUDE.md`, "Ocean column"),
and it is read as an environment variable for the life of one command:
**never written into this repository, never committed, never left on a
rented box** (`ml/CLAUDE.md` §6 — the rule that nothing outliving a job may
be stored on someone else's machine applies here unchanged).

If your account turns out to sit under an organization after all, this is
the step that will refuse. See §0.

---

## 6 · The arithmetic, before anything is provisioned

This is the part worth reading twice, because the honest answer is *we do
not know yet*, and the setup above is what buys the measurement.

**What a TPU costs.** Published on-demand list price is **$1.20 per
chip-hour** for v5e in `us-central1`, `us-east5`, `us-west1` and `us-west4`
(higher in `us-south1`, `europe-west4` and `asia-southeast1`), and **$2.70
per chip-hour** for Trillium/v6e. A **v5e-8 is 8 chips**, so on demand it is
**about $9.60/hour**, and a v6e-8 about $21.60/hour. Spot prices are lower
and are **dynamic** — Google republishes them up to once every 30 days — so
the only honest spot number is the one the console shows you on the day.

**What the current fleet costs.** The rented GPU boxes run **$0.30–0.94/h**,
and a 200k-step stage-2 run takes **15–36 h**, i.e. roughly **$5–30 per
run** (`ml/CLAUDE.md` §3b(c) prices one at 15.6 h and ~$4.6).

**So the bar is throughput, not price.** An on-demand v5e-8 is ~10–30× the
hourly rate of a Vast box; it has to be that much faster before it is
cheaper, and spot narrows the gap without closing it by itself. That is a
plausible bet for a transformer workload on hardware built for one — it is
not a settled one, and it is not what the setup is *for*. The setup is for
measuring it.

**The specific thing that could make it disappointing** is already named in
`ml/plans/JAX_PORT.md` §1b: the per-batch gather (`LazyPixels`/`gather_px`)
runs on the TPU VM's host CPU, and if it cannot feed the accelerator the
chips idle at full price. The first smoke run measures step time and host
utilisation before anything larger is provisioned. If the gather starves the
TPU, the fix is a pre-gathered shard format — a build-side change, not a
model change — and it should be made because a measurement asked for it.

**So: one v5e-8, one smoke run, one measurement.** Do not reserve a large
slice on the strength of a hoped-for speedup.

---

## 7 · What I need from you, in one list

1. The **project ID**.
2. The **zone** you were granted quota in, and which quota (spot or
   on-demand, v5e or v6e).
3. The **bucket name**.
4. The **service-account JSON key**, pasted into chat.

With those four, the next steps are mine: stage a tensor into the bucket,
bring up a v5e-8, run the JAX stage-1 trainer on a toy budget, and report
step time against the fleet's measured pace. Nothing on this list blocks the
JAX work that is still outstanding — the trainers (tier 3) are being built
against CPU parity first, which is the right order regardless of what the
hardware turns out to cost.

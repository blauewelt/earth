# Infrastructure design — the ML training fleet

Chris, 2026-08-10: *"it seems we need to write an infrastructure design doc at
some point, as it is the infrastructure that has bitten us a lot."*

He is right, and the record is unambiguous. On 2026-08-09 a single session
spent roughly six hours and a dozen dispatches and produced **one** scientific
result. Almost every hour lost went to infrastructure, and — this is the part
worth designing against — **almost none of it announced itself**. Steps
reported success while doing nothing. This document describes what exists, the
failure taxonomy it has actually produced, and the invariants that follow.

It is a companion to CLAUDE.md §6c (working principles) and §6d (security
posture), not a replacement: this file is the topology and the failure
catalogue, those are the rules.

---

## 1 · What exists

Four storage layers, three compute locations, one control plane.

**Compute.** GitHub Actions is the control plane; the work runs on three
rented RTX 4090s from Vast.ai, registered as self-hosted runners
(`gpu-box-*`, label `gpu`). A job is a `workflow_dispatch` of
`.github/workflows/ml-train.yml`. There are no other triggers on that
workflow, and that is load-bearing (see §5). GitHub-hosted `ubuntu-latest` is
available for small CPU work but is a trap for training — see the `runner`
input's comment.

**Storage, in increasing order of durability:**

| layer | lives as long as | holds | reachable from |
|---|---|---|---|
| job workspace | one job | everything the job makes, until `actions/checkout` wipes it | that job only |
| `/opt/earth-cache` on a box | the box | data cache (~11 GB), codec + head mirrors | jobs on that box |
| Actions artifacts | 30 days | `pixelmae-<n>`, `probes-<n>` | anyone with repo access |
| GitHub releases | forever | `data-cache-v1`, `model-checkpoints-v1` | anyone, no auth |

**Telemetry.** Three channels, all file-based, all read by a credential-free
status page:
- `ml-live-<n>` — an orphan branch per run, force-pushed every 5 minutes with
  `metrics.jsonl` + `phase.json`. Deleted when the run ends.
- `ml-metrics` — the permanent archive: `run-<n>.jsonl` per run, plus
  `fleet.json` (balance, burn rate, box count — never credentials).
- `ml/EXPERIMENTS.md` — what each run was FOR. The status page links run
  numbers to it by experiment ID.

**Why this shape.** The status page is deliberately credential-free, so every
signal it needs must land in a public file rather than behind an API call. And
the GPU boxes are rented from strangers, so nothing that outlives a job may be
stored on them (CLAUDE.md §6d).

---

## 2 · The failure taxonomy

Every one of these actually happened. They are grouped by mechanism, because
the mechanism is what a design can address.

### 2a · Silent success — the dominant class

A step exits 0 having done nothing. Four distinct instances in one day:

- `"$TAG__pixelmae.pt"` expands the variable `$TAG__pixelmae` (underscores are
  legal in a bash identifier), so the release-asset pattern was a bare `".pt"`.
- The same step called `gh`, which is **not installed on the Vast boxes** —
  127 in under a millisecond, stderr discarded by `2>/dev/null`, step green.
  Between these two, the checkpoint seed had *never once worked*, so the
  box-local resume livelock it was written to cure was still live and killed
  three runs.
- `publish_live_metrics.sh` with no `GITHUB_TOKEN` in the step's `env:` —
  pushes as nobody, fails, swallowed by the `|| true` that exists so telemetry
  cannot kill training. Found in the joint step, then again in the probes step.
- A dispatch that omitted `runner` fell back to `ubuntu-latest`: a 4-core CPU
  box, CPU torch wheel, and nothing in the run's own output saying so.

**Design response.** Best-effort is a promise about *delivery*, never about
*reporting*. Any `|| true` / `2>/dev/null` / `continue-on-error` must print why
it gave up. Assert the **effect**, not the invocation — the seed step logged
"trying release asset …" for three runs while downloading nothing.

### 2b · Work that exists in exactly one place

- Codec checkpoints were lost to cancellation until a pre-checkout rescue step
  was added (#62).
- The **stage-2 head was never protected at all**: `temporal.pt` is written to
  the workspace and uploaded by a step that runs *after* the entire probe
  ladder, so a job that hits its timeout loses every step of it. #112 spent
  nine hours in that state before this was noticed. Now mirrored to
  `/opt/earth-cache/ckpt/<CKPT_TAG>-temporal.pt` on the metrics cadence
  (atomic `.part` + `os.replace`) and swept by the pre-checkout rescue.
- `--require-resume` exits before the eval that writes `pixelmae.pt`, so a
  resume-only job could leave the next step with no codec.

**Design response.** Anything expensive must exist in two places within
minutes of being made, and the second place must not be the machine that made
it. Write atomically; a mirror that can be truncated is not a mirror.

### 2c · Limits treated as physics

- `job_timeout` defaults to 350 minutes, and a 60,000-step stage 2 needs ~7.9
  hours. #111 was **cancelled** over this before anyone read the input's own
  description: *"self-hosted can run for days."* It was a default, not a cap.
- `workflow_dispatch` genuinely caps at 25 inputs, and a 26th makes the file
  unparseable so that *every* dispatch 422s. That one is real, and is why
  knobs get encoded into existing inputs (`window` carries the loss mode,
  `lr_floor` carries the joint codec LR).

**Design response.** Before working around a limit, read whether it is one.
Before dispatching a long job, size it against its own timeout — at the
measured 0.419 s/step, 24,000 stage-2 steps fits in 350 minutes and 60,000
does not.

### 2d · Instruments blind to the failure they exist to catch

- `r_rec_probe` (reconstruction on a fixed batch) looked healthy through a 40×
  embedding collapse, because the cheat does not touch reconstruction.
- `z_shrink` was then added to catch exactly that — and coloured red only
  *above* 1.2, so the next failure, a 250× **expansion**, rendered in grey as
  "0.00x" and looked ordinary. A one-sided guard on a two-sided quantity.

**Design response.** Ask *what would look identical whether this works or
fails*, and measure that. Guard both directions of any ratio. Prefer
invariants with an exact expected value (the twin head's "r_fore must read
1.000000 at step 1, or exit") over thresholds — thresholds are the tripwires
that killed two healthy runs.

### 2e · Scheduling and state

- A queued Actions job can wedge against runners GitHub itself reports online
  and idle. Cancel and re-dispatch; both picked up within 90 seconds. Do not
  restart the boxes — that costs the warm caches for nothing.
- Run-level status lags the jobs endpoint by minutes; `runner_name` on
  `/actions/runs/<id>/jobs` is the truth.
- Vast runner names are **stale instance IDs** from whenever the box was
  created and do not match current instance IDs; every box carries the same
  `label: earth-runner`. The mapping exists only inside each instance's
  `onstart` script, in the runner's `--name` flag. Recover it before stopping
  anything, or you stop the box that is mid-run.
- A scheduled shutdown task that said "stop every box even if the harvest
  fails" would have killed #112 nine hours in. **Automation that stops things
  must check what is running first.**

---

## 3 · Invariants

These are the properties the system should hold. Each one is here because its
violation cost something.

1. Nothing expensive lives in one place for longer than one metrics interval.
2. Every best-effort path reports its own failure on stderr.
3. Every long-running step publishes progress while it runs, not at the end.
4. Every dispatch states what it is (`doc`, with an experiment ID) and what
   code it ran (`head_sha`, surfaced by the status page).
5. No credential that outlives a job is ever on a box.
6. `ml-train.yml` is `workflow_dispatch`-only, forever (§5).
7. A job's result must be recoverable from the release without a GPU.
8. Automation that destroys or stops resources checks for running work first.

---

## 4 · Re-running an evaluation without a training run

This is invariant 7 made concrete, and it is what makes a timed-out job an
inconvenience rather than a loss.

```bash
node scripts/fetch_ckpt.mjs                     # list every stored checkpoint
node scripts/fetch_ckpt.mjs f3_anchor41M__pixelmae.pt ml/runs/rerun/pixelmae.pt
node scripts/fetch_ckpt.mjs f3_s2_24k__temporal.pt    ml/runs/rerun/temporal.pt
```

The release is public, so this needs no authentication and no Actions run. The
tensor comes the same way — `base025_na.npz` is a `data-cache-v1` asset that
`build_family3.py` will fetch on its own. With codec, head and tensor in hand,
the probe ladder runs standalone:

```bash
python3 ml/probe_kfold.py  --runs rerun      # the number we argue from
python3 ml/dip_check.py    --run  rerun
python3 ml/temporal.py     --run  rerun --K 24 --steps 0   # eval only
```

**Publish a checkpoint the moment it is interesting**, not when it is
convenient — `node /tmp/upload_ckpt.mjs <file> <name>` in the session, or the
release-seeding step in the workflow.

---

## 5 · The security line that must not move

Self-hosted runners on a **public** repository are the classic dangerous
configuration: a fork's pull request can execute arbitrary code on your
hardware. This fleet is safe only because `ml-train.yml` is
`workflow_dispatch`-only and dispatch requires repo write access. `pages.yml`
does carry `pull_request`, but runs on `ubuntu-latest`.

**Never add `pull_request`, `issue_comment`, `workflow_run` or `schedule` to
ml-train.yml, and never point another triggered workflow at `runs-on: gpu`.**
Such a change looks harmless in review and would hand code execution on the
boxes, plus the job token, to anyone who opens a PR. Full posture in
CLAUDE.md §6d.

---

## 6 · Known gaps

- No `--resume-temporal`: a stage-2 head can now be *preserved* but not
  *continued* — no optimiser state is saved, so a continuation restarts Adam
  and the LR schedule.
- Runners are not `--ephemeral`, so the registration on each box is long-lived.
- `probe_sequence.json` returns all-NaN across the K sweep on recent runs while
  `probe_kfold` is clean; the sequence probe has a bug of its own.
- `make_table.py` no longer reproduces the paper's `chan%` column for
  `f3-anchor41M` — the source JSON is not on disk — so that caption's "generated,
  not transcribed" is currently untrue for one cell.
- Two boxes sit at 44–45 of 50 GB. A full disk takes a runner offline.

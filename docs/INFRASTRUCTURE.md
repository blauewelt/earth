# Infrastructure design — the ML training fleet

Chris, 2026-08-10: *"it seems we need to write an infrastructure design doc at
some point, as it is the infrastructure that has bitten us a lot."*

He is right, and the record is unambiguous. On 2026-08-09 a single session
spent roughly six hours and a dozen dispatches and produced **one** scientific
result. Almost every hour lost went to infrastructure, and — this is the part
worth designing against — **almost none of it announced itself**. Steps
reported success while doing nothing. This document describes what exists, the
failure taxonomy it has actually produced, and the invariants that follow.

It is a companion to `ml/CLAUDE.md` (the rules for ML work — working
principles, dispatch discipline, security posture) and `docs/ML_BASICS.md`
(what the system is and what its numbers mean). This file is the topology and
the failure catalogue; those are the rules and the science.

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
| GitHub releases | forever | `data-cache-v1`, `model-checkpoints-v1`, `embed-cache-v1` | anyone, no auth |

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

### 2c-bis · Guards sized smaller than the thing they guard

The stage-2 embedding cache is **one allocation of 10.4 GiB** — 516 months ×
84,405 ocean pixels × 64 dims × 4 bytes, and every factor is known before the
write begins. The hygiene step in `ml-train.yml` pruned when free disk fell
below **8 GB**. A threshold under the size of the single allocation it exists
to protect cannot ever fire in time: it passes, and then the write fails.

What made it expensive rather than merely wrong is that `open_memmap(mode="w+")`
creates a **sparse** file. The 10.4 GiB is claimed immediately and allocated
page by page across the fifty minutes the embedding takes, so there is no early
failure to notice — the box dies on a write with the cache nearly complete. On
a 50 GB box a full disk also takes the *runner* offline, so one bad run becomes
a machine that eats every job dispatched to it afterwards. #117 cleared the
check with ~11 GB free and spent an hour walking into a wall; it was cancelled
at 45/50 GB, five gigabytes from the end of an embedding needing six.

Underneath the badly-sized guard was a worse assumption. The memmap exists
because a 10.4 GiB Z beside a 10.1 GiB tensor OOM-killed a **7 GB** box twice
on 2026-08-07. The boxes rented since have **126 GB of RAM** and a 50 GB disk.
The constraint moved and the code did not, so the run was writing to the scarce
resource because the abundant one used to be scarce.

**Design response.** A guard's threshold must be derived from the allocation it
guards, not chosen. Where the size is computable — and here every factor was —
compute it, and put the check where those numbers are in scope rather than in a
shell step that can only guess. Then let the check *choose a resource* instead
of only permitting or refusing one: `temporal.py` now takes the disk if the
cache fits (worth ~50 minutes to the next run on that box), RAM if it does not,
and refuses with both numbers when neither works. Scale the headroom to the
write, too — a flat 3 GiB reserve reads as prudence until it refuses a 5 MB
cache on a small sandbox, at which point it is clear the constant was doing the
refusing rather than the risk.

### 2c-ter · Displays that render on an unreachable branch

The planned-schedule preview was written so a resumed cosine could be checked
*before* it spent sixteen hours. It was correct, it was published, it was
publicly readable — and the dashboard showed nothing, repeatedly, because it
was drawn only on the code path where the live branch has **no metrics file at
all**. `temporal.py` publishes its `config` line within seconds of starting, so
that path is live for a few seconds of a sixteen-hour run and false every time
a human looks.

**Design response.** When a feature exists for a window of time, check that the
condition it renders under actually holds *during that window*, not merely at
some instant. The test that accompanied the feature used a queued run with no
branch — it exercised the one arm that already worked. A test fixture must
model the state the user will be in, which here means: in progress, a metrics
file carrying config and nothing else, and a published plan.

### 2c-quater · Guards that are right, and in the wrong place

Run #119 embedded for **93 minutes** and then exited one second later:

> `f3_s2_60k__temporal.pt` predates optimiser-state saving (missing
> `['opt', 'sched', 'step']`) … a warm restart wearing a continuation's name.
> Refusing.

Nothing about that refusal is wrong. Loading weights alone resets Adam's
moments and the LR schedule, and reporting the result as a continuation would
have been a false claim about a published number. The guard is one of the good
ones.

Its POSITION cost an hour and a half of a rented 4090. The checkpoint's keys
are readable the moment the file is on disk — before the tensor is loaded,
before a single encoder forward. The precondition depended only on the inputs,
and it was checked at the point of *use* instead of the point of *dispatch*.

There is a second, quieter cost. Because the job "completed successfully"
(the step was `continue-on-error`), nothing in the run's status said the
experiment had not run. It took reading the log to discover that sixteen hours
of training had become two hours of nothing.

**Design response.** A precondition that depends only on the inputs must be
checked while the inputs are all it has cost you.
`scripts/precheck_stage2_head.py` answers it in under a second and names the
flag that would have worked. Two refinements learned immediately afterwards,
both worth keeping:

- **Only a DEFINITE answer may be fatal.** GitHub runs `run:` blocks under
  `bash -e`, so a non-zero exit from the precheck aborts the whole step and
  takes the k-fold ladder, `dip_check` and both head probes with it. That is
  correct for "this mode cannot work"; it is wrong for a torch quirk, where
  the behaviour before the guard existed was strictly better. A check that can
  cost more than the thing it protects is the same error one level up.
- **Verify the artefact, not the intention.** The reason this was not caught
  when E-008 was designed is that the entry was written from what the
  checkpoint was *assumed* to contain. Opening the file would have taken ten
  seconds. Every published head turned out to be `{args, model}`.

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

### 2d-bis · Silent divergence between boxes

Each box builds and caches its own tensor. On 2026-08-10 two of them had
drifted: the channel-space persistence baseline — a **data-only** quantity
that cannot depend on the model — read `1.2046650648117065` on
gpu-box-45318655 (runs #88 and #112, bit-identical) and `1.1540812253952026`
on gpu-box-35586926 (#110). It tracked the box, not the run. 4.2% apart,
~350,000× float32 epsilon.

Nothing in any run's output said which tensor it used, so cross-box results
were being compared as though they shared a dataset. Now every run's
`provenance.json` carries a sha256 of the tensor file. **Design response:**
if a computation is supposed to be reproducible across machines, fingerprint
its inputs and publish the fingerprint — an invariant that is never checked is
a hope.

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

### 2f · Work that runs on the wrong hardware

Four scripts in the probe ladder embedded on the CPU beside an idle 4090:
`dip_check.py`, `rollout.py`, standalone `trainprobe.py`, and `probe_head.py`'s
`fold_fit` — the last of which trained its read-out for 4,000 steps per fold on
CPU while the *codec* sat correctly on the GPU. The pattern is identical every
time: someone moves the codec and stops there, because the codec is the big
model and it feels like the whole job.

`embed_everything` follows **the model's** device, by design and by docstring.
So does every torch module. **Design response:** when a script gains a second
model, ask where that one runs. And watch the one signal that shows this —
`gpu_util` — because nothing else does: the run reads "in progress", the runner
"online", the box "running", and the job is eight hours of CPU.

**The fifth and largest instance was not an oversight — it was a decision whose
premise expired.** `temporal.py` sent the codec to the accelerator for the
embedding and brought everything back, with a comment saying so deliberately:
*"stage-2 training is a small transformer for a few thousand steps"*. That was
true when stage 2 was 4,000 steps. It is now 140,000 and 200,000, and the
comment was still there, still reasonable-sounding, protecting a choice that
had stopped being right.

Measured before changing it: at batch 256, K=24 and a 1.824M head, the data
gather off the memmap is **12.4 ms of a 725 ms step** — the model is 98% of
the cost. So a 200,000-step run was a full day of a rented 4090's CPU while its
GPU sat at zero.

This is the same shape as §2c-bis, where the embedding memmapped to a 50 GB
disk because the box used to have 7 GB of RAM. **A documented decision is
harder to re-examine than an undocumented one**: the comment answers "why is
this on the CPU?" so convincingly that nobody asks whether its premise still
holds. When a scale changes by 50x — steps, RAM, disk, dataset — go and reread
the comments that mention the old scale. They will still sound right.

### 2g · Expensive derived work with no home

Three storage layers existed for things the fleet makes — tensors, codec
checkpoints, stage-2 heads — and the single most expensive derived artefact had
none of them. `Z`, the frozen-codec embedding, is ~95 minutes of an RTX 4090
and is IDENTICAL for every run that freezes the same codec. It lived on one
rented box's disk, where the hygiene step deleted it under pressure. #112,
#117, #119, #120 and #121 each rebuilt the same array.

Worse, deleting it was rational at the time: it was the biggest reclaimable
file on a 50 GB disk, so a blanket prune always reached for it first — and the
rebuild is another 10.4 GiB, which puts the box straight back under the
threshold. A treadmill, not a one-off.

**Design response.** Three changes, each closing a different part of it:

- `embed-cache-v1`, keyed by the **codec's weight hash** so a cache can never
  be applied to a codec that did not produce it, chunked at 1.5 GiB under
  GitHub's asset cap, and **verified on pull** by length and dtype (a file that
  is too LONG — a chunk uploaded twice or reassembled out of order — maps
  cleanly and returns real numbers for the wrong months).
- **float16**, halving it to 5.2 GiB so it fits on the hardware we can actually
  rent, at a measured cost of ~1e-7 of the quantity being reported.
- The prune **keeps the newest** cache instead of deleting all of them, and
  `temporal.py` prunes only stale hashes, which it can identify and a shell
  step cannot.

### 2h · Long work with no resume, and no way to see how far it got

The embedding is the longest single phase of a stage-2 run and, until
2026-08-10, it (a) reported nothing between "starting" and "done", and (b)
could not be resumed. Actions will not serve logs for a running job, so the
only way to tell "working" from "wedged" for ninety-five minutes was to watch
the box's resident memory climb — a thermometer taped to the outside of the
oven. And a crash at 80% cost the whole ninety-five minutes.

**Design response.** It reports every 5% (month, elapsed, ETA, and whether Z is
going to disk or RAM) into the run's own `metrics.jsonl`, which the publisher
pushes to `ml-live-<n>` — the side channel is the only one readable while it
matters — and the status page renders it as a progress bar. It resumes from a
marker written **after** each flush, so the marker can only under-claim; an
over-claiming marker would skip months holding zeros, which is a wrong answer
with no symptom.

---

## 3 · The monitor

`node scripts/fleet_health.mjs` — one command, one verdict, exit non-zero when
unhealthy. Chris asked for this after the eight-hour CPU tail: *"having a
monitor check every 30 mins on whether the gpu is idle would be very helpful."*
Four checks, ordered by what each has cost:

| check | condition | why it exists |
|---|---|---|
| **CPU-BOUND** | a job running with `gpu_util` < 5% and `cpu_util` > 20% | §2f — eight hours, four scripts |
| **IDLE BURN** | a box `running` with no job on it | paying for nothing; happened repeatedly |
| **QUEUE STALL** | a job `queued` while a runner is online and idle | §2e — 22 minutes, twice |
| **DISK** | over 90% | a full disk takes a runner offline |

It runs hourly as a scheduled task (the platform's floor is hourly, not the 30
minutes asked for; two offset hourly tasks would give 30-minute coverage if
that turns out to matter). It reports only when something is wrong.

One legitimate CPU-bound case to know: the k-fold ridge solves are numpy and
genuinely CPU work. They are minutes. Sustained CPU-only for half an hour or
more means a missing device move.

One false positive to know: Vast's `/instances/` intermittently serves a
**dead GPU frame** — `gpu_util`, `gpu_temp` and `vmem_usage` all exactly 0 —
for a box that is mid-training (first seen 2026-08-15 on run #310: dead frames
alternated with 89–100% / 69–75°C / 16 GB frames ~20 s apart). `cpu_util` is
host-side and stays real, so a dead frame satisfies the CPU-BOUND condition
while the job trains at full speed. A real §2f failure still produces live
frames (an idle GPU reads ~30–50 °C, never 0 °C). The monitor now resamples
for up to a minute and judges only live frames; if every frame stays dead it
reports `TELEMETRY` instead of CPU-BOUND. Ground truth either way is the
run's `ml-live-<n>` branch: a training job appends `stage2_wall_s` every
~5 min, and a step rate matching its sibling boxes cannot be a CPU run.

## 4 · Invariants

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
9. Every model in a script runs on the same device as the data it consumes —
   and `gpu_util` is watched, because it is the only signal that says otherwise.
10. Any state a run needs to be CONTINUED (optimiser moments, schedule
    position, RNG) is saved with the weights, and a test asserts the
    continuation matches an uninterrupted run.
11. Every resource guard is sized from the allocation it guards, computed
    where the numbers are in scope, and chooses between resources rather than
    only permitting or refusing one.
12. A run reports the SPACE it will need before it starts consuming it, and a
    result file is never written containing NaN — the job stops instead, so a
    fault is attributed to its cause rather than to the last thing that
    touched it.
13. Any artefact costing more than ~10 minutes to produce leaves the box that
    made it, keyed by something that identifies WHAT it is (a content or
    weight hash), and is verified before it is trusted.
14. Long work reports its own progress, and resumes where it stopped. The
    progress marker is written after the data it describes, never before.
15. A shared artefact is published when it EXISTS, not when the job that made
    it ends — otherwise dependents cannot be scheduled against it.
16. A precondition that depends only on the inputs is checked at dispatch, and
    only a definite negative may be fatal.

---

## 5 · Re-running an evaluation without a training run

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

## 6 · The security line that must not move

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

## 6b · Two design changes agreed 2026-08-10, not yet built

Both came from Chris after watching E-008 cost three dispatches. Neither is
urgent; both remove a whole class of problem rather than an instance.

### Split the pipeline into three jobs

Today one job builds the tensor, embeds, trains stage 2 and runs the probe
ladder. The embedding is the odd one out: it is **shared, deterministic and
expensive** — ~95 minutes for any run that freezes the same codec — while
everything around it is per-experiment. Bundling it means every experiment
pays for it, and two experiments started together pay twice for the same
bytes, which is exactly what #120 and #121 did.

    job 1  EMBED    build (or pull) Z for a codec hash, publish it, exit
    job 2  TRAIN A  pull Z, train the 140k warm restart, probe
    job 3  TRAIN B  pull Z, train the 200k from scratch, probe

Job 1 runs once per codec, ever. Jobs 2 and 3 start when it finishes and take
minutes to reach their first training step instead of ninety-five. The
dependency is real work, not scheduling politeness: they cannot start earlier
in any useful sense, because there is nothing to pull until job 1 publishes.

Note what had to be fixed before this decomposition could pay off at all: the
cache push used to sit after `wait $S2_PID`, i.e. *after training*, so
"publish then dependents pull" described nothing that happened. It now pushes
the moment the file exists. A dependency graph over jobs that do not publish
until they finish is just a slower sequence.

### Stop baking the step budget into the learning rate

`CosineAnnealingLR(T_max=steps)` makes the rate a function of the total, and
that single choice produced two separate problems:

- **The bug.** A checkpoint's schedule is only meaningful beside the budget it
  was trained under. Reload it while asking for a larger total and it believes
  it is finished: `lr = 0.0`, sixteen hours of updating nothing, every status
  reading success. Containing it took a rebuild-the-schedule special case, a
  refusal guard and a test.
- **The comparability tax.** A 6,000-step run and a 200,000-step run are at
  different rates at every shared step, so they are two experiments that share
  an architecture rather than a prefix and its extension. E-007's three points
  each had to be described as "its own converged cosine", and the 200k point
  could not be a continuation of the 60k one — which is the entire reason
  E-008 became a warm restart.

`--lr-schedule invsqrt` (implemented, not yet default) makes `lr(s)` a pure
function of `s`. Resume stops being a case at all; two budgets become a prefix
and its extension; checkpoints are interchangeable.

The honest cost, because it is a trade and not a free win: cosine anneals to
zero and therefore *converges* at a known point, which is what makes "the 60k
result" a settled number. invsqrt never reaches zero, so results are "at step
N" rather than "converged". For a programme whose question is "does more
compute help?" that is the better shape — the question presumes an open-ended
curve — but the switch should be its own experiment (one budget, both
schedules), not a default flipped in passing.

---

## 7 · Known gaps

- ~~No `--resume-temporal`~~ — **closed 2026-08-10.** The head checkpoint now
  carries optimiser, scheduler, step and RNG state, `--resume-temporal` loads
  all of it, and `tests/test_resume_temporal.py` asserts that N + resume + N
  lands exactly where 2N does (max|Δw| = 0) *and* that dropping any one saved
  piece makes it diverge, so the test cannot rot into a tautology. The workflow
  reaches it through `window: resume2:<tag>`.
- Runners are not `--ephemeral`, so the registration on each box is long-lived.
- `probe_sequence.json` returns all-NaN across the K sweep on recent runs while
  `probe_kfold` is clean; the sequence probe has a bug of its own.
- `make_table.py` no longer reproduces the paper's `chan%` column for
  `f3-anchor41M` — the source JSON is not on disk — so that caption's "generated,
  not transcribed" is currently untrue for one cell.
- Two boxes sit at 44–45 of 50 GB. A full disk takes a runner offline.

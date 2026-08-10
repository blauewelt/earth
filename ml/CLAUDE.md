# earth / ml — standing instructions for the training & research work

This file governs everything under `ml/`, `scripts/` that touches the fleet,
and `.github/workflows/ml-train.yml`. The repository root `CLAUDE.md` governs
the globe app (layers, tiles, UI, deploys) and its rules — "deploy first",
"stamp before you commit", the layer checklist — **do not apply here.** They
were written for a static site where a bad deploy is reversible in ninety
seconds. A bad dispatch here costs hours of rented GPU and, worse, can produce
a number that looks like a result.

Chris asked for the split on 2026-08-10, after a day in which the two kinds of
work interleaved badly.

Read alongside:

| file | for |
|---|---|
| `docs/ML_BASICS.md` | what the system IS and what its numbers mean |
| `docs/INFRASTRUCTURE.md` | the fleet, its failure taxonomy, its invariants |
| `ml/EXPERIMENTS.md` | what was run and what it returned |

---

## 0 · The three rules that would have saved the most time

Everything below elaborates these.

1. **Verify the ARTEFACT, not the intention.** #119 spent 93 minutes embedding
   and then refused, because the checkpoint it was told to continue contained
   no optimiser state. The design had been written from what the file was
   *assumed* to hold. Opening it would have taken ten seconds.
2. **A step that reports success is not evidence it did anything.** Assert the
   effect. Four separate steps in one night reported success while doing
   nothing.
3. **Check a precondition where the inputs are all it has cost you.** A guard
   at the point of USE is correct and expensive; the same guard at dispatch is
   correct and free.

---

## 1 · Before you dispatch

- **State what result would FALSIFY the hypothesis**, and check the
  configuration can produce it. Several runs were healthy and could not, by
  construction, test their own hypothesis.
- **Write the EXPERIMENTS.md entry at dispatch**, hypothesis first, so the log
  cannot be rewritten to fit the answer. Name the control explicitly.
- **Open every checkpoint the dispatch names** (`scripts/precheck_stage2_head.py`
  does this for stage-2 heads) and confirm it supports the mode you are asking
  for.
- **Size the job against its own timeout.** Measured: 0.419 s/step for a 192×4
  head at K=24. So 24,000 steps fits in 350 minutes and 60,000 does not, and
  200,000 needs ~23.3 h plus the embedding. `job_timeout` is an INPUT, not a
  cap — self-hosted runners can run for days.
- **Publish the plan** (`scripts/publish_plan.mjs <n> '<json>'`) so the status
  page can draw the schedule before the run spends anything.
- **`runner` defaults to `gpu`.** Omitting it used to send jobs to a free
  4-core CPU box where a 40M codec "trains" and nothing says wrong-hardware.
  Check `runner_name` on any dispatch you care about.

## 2 · While it runs

- **VERIFY IN THE FIRST MINUTES.** For a resumed or restarted run, check the
  learning rate is what the plan says and is not zero. Cancel if it reads
  otherwise; an hour of a wrong run costs more than a re-dispatch.
- **A queued job against an idle runner is stuck, not slow.** Cancel and
  re-dispatch (this has worked within 90 seconds). Do not restart the boxes
  and lose their warm caches. If a re-dispatch also stalls, stop/start the box
  — that clears a wedged runner and keeps the disk.
- **Watch `gpu_util`.** It is the only signal that says "this is running on the
  wrong device" — four eval scripts silently embedded a 40.7M codec on CPU.
- **Report measurements, not intentions.** "Curves will appear shortly" said
  twice without reading the branch is a guess wearing a fact's clothes.
- **SCREENSHOT THE STATUS PAGE at every monitoring wake-up**, and look at it:
  `node scripts/status_shot.mjs --out /tmp/status.png`. It captures the live
  branches and plans and renders the deployed page at a phone viewport, so
  what you check is what Chris sees. Standing rule, requested 2026-08-10 after
  I twice described a dashboard state that was not real — once because the
  plan preview rendered on a branch nobody is ever on, once because a stale
  record from a previous run was being charted as the current one. Both times
  I reported what the code should produce instead of what the page produces. A
  screenshot is the one check that cannot be fooled by my model of the code.
  It also prints PAGE ERRORS, which is how a blanked section gets noticed
  before a human finds it.

## 3 · Reporting a result

- **Numbers come from `probe_kfold.py`** — the year-blocked k-fold. Anything
  else (in-training light probe, a 36-month split) must be labelled as such.
- **A number without its baseline is not a result.** Wind-only ridge: 0.531 at
  1°, 0.568 at quarter degree.
- **Comparing two probes needs a PAIRED test** (`scripts/paired_probe.py`), not
  two overlapping intervals — they share folds, months and most of their error.
- **Record negative and null results with the same care as positive ones.**
  Most entries in EXPERIMENTS.md are null and they are why the programme knows
  where its bottleneck is.
- **Record the COST too.** E-008 spent three dead dispatches and ~2.5 h of GPU
  before a single training step; an entry showing only the successful run makes
  the answer look cheaper than it was.
- **A rerun is not a resample.** Before dispatching a repeat, check the code
  has a knob for the thing you intend to vary.

---

## 4 · Working principles (written after the night of 2026-08-09)

That session burned about six hours and a dozen dispatches and produced one
scientific result. Everything else was self-inflicted. These are the rules
extracted from it; they are ordered by how much they would have saved.

**On building.**

1. **Prefer the formulation that removes a failure mode over the one that
   guards against it.** The joint loss got three revisions of increasing
   cleverness — a step-0 constant, a scale-free ratio, a twin reference
   trained alongside — each *policing* a degeneracy that a different choice of
   measurement space *abolishes*. When you are adding a correction to a
   correction, the earlier choice is wrong, not under-specified. Stop and
   change it.
2. **Normalise by properties of the DATA, never by properties of the MODEL.**
   A denominator the model can influence is a term in the objective, and it
   will be optimised. Variance of an observed field is fixed and ungameable;
   a frozen checkpoint's loss is not, once what was frozen is what you are
   training. This one sentence covers the shrinkage degeneracy, the detached
   denominator, and both circular reference constants.
3. **Keep diagnostics out of the objective.** "Am I still as good as the model
   I started from?" is a question to LOG, not to optimise. Smuggling it into
   the loss is what made one experiment depend on a constant hand-copied from
   another, and then on that other experiment's arbitrary stopping point.
4. **Sequenced experiments are a smell.** If B needs a number from A, ask
   whether B can measure it itself. Interdependence costs A's wall-clock plus
   a constant that silently goes stale.
5. **For a change to an objective, write the algebra before the code.** The
   detached-denominator bug was a two-line sympy check away, and that check
   is what finally settled it. For a loss, the gradient IS the specification;
   the code is a transcription.

**On verifying.**

6. **A step that can fail silently will.** Every `|| true`, `2>/dev/null` and
   `continue-on-error` must say why it gave up. Best-effort is a promise about
   *delivery*, not about *reporting*. Four separate steps in one night
   reported success while doing nothing: a release seed with a mis-braced
   variable, the same seed calling a CLI the boxes do not have, a live-metrics
   publisher with no token, and a dispatch that silently chose a CPU box.
7. **Assert the EFFECT, not the invocation.** The seed step printed "trying
   release asset …" and downloaded nothing for three runs. A log line proves a
   line of code ran; it proves nothing about the world.
8. **Exercise the code path on a toy before spending the expensive
   resource.** A synthetic 8×9×5 tensor and a 1-layer codec took five minutes
   and exercised three loss modes end to end. Any hour of GPU on a path that
   has never executed is a coin flip. `ml/train_joint.py --smoke` against a
   generated npz is the pattern.
9. **Build an invariant with an EXACT expected value and make the job refuse
   if it fails.** "r_fore must read exactly 1.000000 at step 1, because both
   branches are identical until the first update" is worth more than any
   amount of careful reading. Prefer exact identities to threshold checks —
   thresholds are the tripwires that killed healthy runs (#86, #91).
9b. **A degeneracy you can NAME is one you must close or measure — never one
    you may rank as improbable.** While designing the scale-free denominator I
    wrote down the second cheat (inflate the persistence baseline instead of
    shrinking z) and judged it "a real possible cheat but far less trivial
    than pure scaling. Worth noting, not worth blocking." It was the dominant
    direction and arrived faster than the one being fixed. If you can describe
    the exploit in a sentence, the model can find it in a thousand steps.
10. **Instrument the quantity that DISTINGUISHES the stories, not the one
    that is easy to plot — and guard BOTH directions.** Reconstruction-on-a-fixed-batch looked healthy
    through a 40× embedding collapse because it was structurally blind to it.
    Ask: *what would look identical whether this works or fails?* — and
    measure that. `z_shrink` turned the next occurrence into a ten-minute
    diagnosis instead of a retracted result — and then failed on its own
    terms: it was written for CONTRACTION and coloured red only above 1.2, so
    the 250x EXPANSION that followed rendered in grey and looked ordinary. A
    one-sided guard on a two-sided quantity is not a guard. Flag on
    |log(ratio)|.

**On deciding and reporting.**

11. **Distinguish "it is running" from "it can answer the question."** Several
    runs were technically healthy and could not, by construction, test their
    own hypothesis: a 95/5 gradient split, a reference that put the forecast
    term permanently below the reconstruction term. Before dispatch, state
    what result would falsify the hypothesis and check the configuration can
    produce it.
12. **Report measurements, not intentions.** "Curves will appear shortly" was
    said twice without reading the branch. An unchecked status claim is a
    guess wearing a fact's clothes.
13. **Prefer one clean run to five relaunches.** Each relaunch was locally
    justified; collectively they meant nothing finished. When the same
    artefact is relaunched twice, stop spending and fix the whole class of
    problem first. An idle GPU at $0.27/h is far cheaper than a confident
    wrong experiment.
14. **A queued job against an idle runner is stuck, not slow.** Cancel and
    re-dispatch (see the queue-stall note in Part 2); do not restart the boxes
    and lose their warm caches.

---

## 5 · Principles added 2026-08-10

The day E-008 took three dispatches to start. Same spirit as §4, different
failures.

15. **Verify the ARTEFACT, not the intention.** The E-008 design said the run
    "reloads the 60k model, optimiser moments and RNG stream". No published
    head has ever carried optimiser state. The claim was written from what the
    checkpoint was assumed to contain, and cost 93 minutes of embedding to
    disprove. Open the file.
16. **A guard can be right and in the wrong place.** The refusal that caught
    it was correct and fired after the expensive part. A precondition that
    depends only on the inputs must be checked while the inputs are all it has
    cost you.
17. **Only a DEFINITE answer may be fatal.** GitHub runs `run:` blocks under
    `bash -e`, so a non-zero exit from a precheck aborts the step and takes the
    whole probe ladder with it. Right for "this cannot work"; wrong for a torch
    hiccup. A check that can cost more than the thing it protects is the same
    error one level up.
18. **Size a guard from the allocation it guards.** The disk check fired below
    8 GB against a single 10.4 GiB write, so it could not fire in time by
    construction. Where the size is computable, compute it — and put the check
    where those numbers are in scope, not in a shell step that can only guess.
19. **Let a check CHOOSE a resource, not merely permit or refuse one.** The
    embedding memmapped to a 50 GB disk on a box with 126 GB of RAM, because
    the code was written for a 7 GB box. When one resource is exhausted and
    another is not, use the other and say so.
20. **Publish a shared artefact when it EXISTS, not when the job ends.** The
    embed-cache push sat after `wait $S2_PID`, i.e. sixteen hours after the
    cache was finished — so "wait for the first job, then start the second"
    described nothing that could happen. A dependency graph over jobs that do
    not publish until they finish is just a slower sequence.
21. **Flush, THEN mark.** A resumable artefact's progress marker must be
    written after the data it describes, so it can only under-claim. An
    over-claiming marker makes the next run skip work that was never done —
    real numbers, wrong months, no symptom.
22. **Never write NaN into a results file.** Stop instead. A results file full
    of NaN is loud enough to notice and quiet enough to misattribute: the
    all-NaN sequence probe was blamed on the probe twice before anyone looked
    at the mask.
23. **A display only helps if its render condition holds during the window it
    is for.** The planned-schedule preview drew only when the live branch had
    no metrics file — true for a few seconds of a sixteen-hour run, false
    every time a human looked. Test the state the user will actually be in.
24. **Do not leave a stale reference table in the log.** A planned-LR table
    that no longer matches the run is worse than none: it gets checked, it
    matches nothing, and the run takes the blame for the document.

---

## 6 · Security posture of the rented GPU boxes

The Vast boxes are strangers' machines. The host has root on the physical
machine, so **treat everything on a box as readable by whoever owns it**.
That single assumption decides everything below.

**What is on a box.** The workflow's automatic `GITHUB_TOKEN` (an env var for
the job's lifetime, repo-scoped, `contents: write` per the `permissions:`
block, revoked by GitHub when the job ends); the Actions runner's registration
credential under `/opt/runner`; and public data plus model checkpoints, none
of it confidential. The runner runs as root inside its container
(`RUNNER_ALLOW_RUNASROOT=1`).

**What is NOT, and must never be.** The user's PAT (`/home/claude/.gh_pat`
lives in the session sandbox), the Vast API key, and the CMEMS credentials.
Nothing that outlives a job and nothing that reaches another repo or account.

**The registration is a bigger exposure than the token.** The job token dies
with the job. The registration persists on disk, survives stop/start, and is
NOT `--ephemeral` — so anyone who lifts it can act as a runner for this repo
indefinitely, receiving jobs and minting a fresh `GITHUB_TOKEN` each time.
Mitigations, cheapest first: pass `--ephemeral` when creating a box (one
registration per job); remove the runners from the repo
(`scripts/fleet_rm_runner.mjs`) when the fleet will be idle for a long stretch
— note this breaks the cheap stop/start loop, since a started box reconnects
with a credential GitHub no longer honours and the onstart script skips
re-registration when `/opt/runner/.runner` exists.

**The one line of defence that must never move.** Self-hosted runners on a
PUBLIC repo are the classic dangerous configuration: a fork's pull request can
execute arbitrary code on your hardware. We are safe only because
`ml-train.yml` is `workflow_dispatch`-only and dispatch requires repo write
access. `pages.yml` does carry `pull_request`, but runs on `ubuntu-latest`.
**Never add `pull_request`, `issue_comment`, `workflow_run` or `schedule` to
ml-train.yml, and never point another triggered workflow at `runs-on: gpu`.**
That change would look harmless in review and would hand code execution on the
boxes, plus the job token, to anyone who opens a PR.

**Blast radius if a job token does leak.** Repo-scoped `contents: write`: push
to any branch including main, delete branches, cut releases. Not other repos,
not the account. Keep `permissions:` at `contents: write` and add nothing;
`contents` is needed only for the `ml-live-*` side channel and the `ml-metrics`
archive.

---

## 7 · Domain lore (hard-won; do not relearn)

### The control plane

- **`workflow_dispatch` allows at most 25 inputs, and a 26th breaks the WHOLE
  workflow.** `ml-train.yml` sits exactly at the ceiling. Exceeding it does not
  fail gracefully — GitHub refuses to parse the file, every dispatch 422s, and
  the Actions list shows a failed run named after the file path with no jobs,
  which reads like a broken runner. Encode new knobs into existing inputs:
  `window` carries the loss mode, `unroll:N`, `resume2:<tag>[@lr]` and
  `warm2:<tag>[@lr]`; `lr_floor` carries the joint codec LR. **Count the inputs
  before pushing a workflow edit.**
- **`runner` defaults to `gpu`.** Omitting it once sent a run to a free
  4-core GitHub box where the install step picks the CPU torch wheel and a 40M
  codec "trains". Nothing in the output says wrong-hardware; the only tell is
  the runner name.
- **A queued job can wedge.** Runs have sat queued for 22 minutes while the
  API reported the runners online and idle. Cancel and re-dispatch — that has
  worked within 90 seconds. Check `runner_name`/`status` on the JOBS endpoint,
  not the run-level status, which lags.
- **`bash -e`** governs every `run:` block, so any non-zero exit aborts the
  rest of that step. This is why a precheck must only fail on a definite
  answer (§5.17).

### The boxes

- **A Vast instance's runner NAME lives in its `onstart` script and nowhere
  else.** The names (`gpu-box-45318655`) are stale instance IDs and do not
  match current IDs; every box carries the same `label: earth-runner`. Recover
  the mapping with `(i.onstart||"").match(/--name[= ]"?([^"\s\\]+)/)` over
  `/api/v1/instances/` before stopping anything, or you will stop the box that
  is mid-run.
- **A Vast disk CANNOT be resized, and the API lies about it.** `PUT
  /instances/<id>/` with `{"disk": 80}` and with `{"disk_space": 80}` both
  return `200 {"success": true}` and change nothing — measured on two stopped
  instances, v0 and v1. `scripts/gpu_box.mjs resize` refuses with that
  measurement rather than issuing a call that appears to work. To get more
  disk, create a box with a larger `DISK` and destroy the old one; the data
  cache re-seeds from `data-cache-v1`.
- **50 GB is a real design constraint**: a ~15 GB torch image, ~11 GB of
  tensors and a 5.2 GiB embedding cache leave little room. At float32 the
  cache did not fit at all, which produced a delete-rebuild-delete treadmill.
- **Stopping is the cheap idle state.** It keeps the disk (data cache,
  checkpoint mirrors) and the runner registration, dropping a box from ~$0.27/h
  to storage only. `start` can return `resources_unavailable` and queue the
  state change — it may come up minutes later, so check before renting more.
- **Nothing that outlives a job may be stored on a box** (§6).

### Failure signatures worth recognising instantly

- **`"$TAG__pixelmae.pt"`** expands the variable `$TAG__pixelmae` — underscores
  are legal in a bash identifier. **Always brace a variable followed by `_`, a
  letter or a digit**: `"${TAG}__pixelmae.pt"`.
- **The boxes have no `gh` CLI.** Fetch release assets with
  `curl -fsSL https://github.com/$REPO/releases/download/$TAG/$ASSET` (public
  repo, no auth).
- **`2>/dev/null` on a command whose failure you are branching on** hides the
  one line that explains the branch.
- **A channel that starts later than the tensor** makes a month-0 mask empty,
  and `mean` over an empty section is NaN, not an error. See
  `docs/ML_BASICS.md` §3.
- **`CosineAnnealingLR.load_state_dict`** restores the parent's `T_max`, so a
  reloaded schedule asked for a larger total returns **lr = 0.0**. See
  `docs/ML_BASICS.md` §9; `--lr-schedule invsqrt` removes the whole class.
- **No stage-2 head published before 2026-08-10 can be continued** — all are
  `{args, model}`. Use `window: warm2:<tag>[@lr]` with `temporal_steps` as the
  EXTRA steps. Snapshots written since carry `opt`/`sched`/`step`.

### Artefacts and where they live

| artefact | where | keyed by |
|---|---|---|
| codec checkpoints | `model-checkpoints-v1` release | tag |
| stage-2 heads | `model-checkpoints-v1`, `run-<n>-temporal-latest.pt` every ~30 min | run number |
| tensors | `data-cache-v1` release, chunked | recipe |
| embedding cache `Z` | `embed-cache-v1` release, 1.5 GiB chunks | **codec weight hash** |
| live metrics | `ml-live-<n>` orphan branch (force-pushed, deleted at end) | run number |
| archived metrics | `ml-metrics` branch (additive) | run number |

---

## 8 · Deferred / open follow-ups

- ~~**Build E-006**~~ — done 2026-08-10: `--loss-mode data` in
  `ml/train_joint.py`, with the gauge invariance checked on the real model
  (`tests/test_e006_gauge.py`: data-space loss ratio 1.000000, z-space
  16.000000 under a ×4 rescale) and an end-to-end CPU smoke
  (`tests/test_e006_smoke.py`). What remains is a DISPATCH decision — budget,
  and the control, which should be the frozen codec rather than another joint
  mode, since the question is whether joint training beats not doing it.
- **Split the pipeline into three jobs** (embed / train A / train B) — designed
  in `docs/INFRASTRUCTURE.md` §6b, now possible because the cache publishes on
  existence.
- **Decide on `--lr-schedule invsqrt`** as the default, via its own experiment
  (one budget, both schedules).
- **Run `scripts/paired_probe.py`** over the head and its raw-3×3 control once
  a run has written the per-month arrays, and only then decide whether the
  +0.034 gap is quotable. Draw a genuine second seed with `--seed-base 3`.
- ~~**Re-score #88/#93 through `probe_kfold`**~~ — **impossible, struck
  2026-08-10.** `probe_kfold` scores the frozen codec, which #88 and #93
  share, so it returns the same number for both by construction. Asking for
  it is what sent E-009 out with an instrument that could not move with its
  own variable. The replacement is `rapid_probe_kfold` (`temporal.json`), and
  E-009's #131–#134 re-measure the whole U axis on it. See
  `docs/ML_BASICS.md` §5b.
- **Unroll U=2 and U=8.**
- **The parameter-bottleneck question** — needs a from-scratch run at larger
  width; deliberately deferred.

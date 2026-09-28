# Rolling fleet-check log

One compact entry per hourly check. **Deliberately one file, not one file per
hour** — ~610 per-hour `fleet-check-*.md` docs are what pushed the project store
over its 2,000,000-token cap (standing item 41). Do not reintroduce that pattern
here. Long-form records only when an hour actually warrants one; the
2026-09-28 03:30Z doc is the current baseline and stays as its own file.

Newest first.

---

## 2026-09-28 11:30Z — HEALTHY (exit 0), fleet unchanged from 10:30Z. 0 mutations, 0 fleet commits.

`fleet_health.mjs`: `1 run(s) not finished · 0 runner(s) online+idle` / HEALTHY / EXIT=0.
Reading at 11:30:15Z, direct `/api/v1/instances/`.

- **CPU-BOUND** n/a — no box `running`, no job anywhere on the fleet. No threshold used,
  no control read, no script to name.
- **IDLE BURN** clean — 4 instances, all `cur_state=stopped`, 0 of 21 runners online.
  **$0.00/h GPU burn.** No `gpu_box.mjs stop` issued.
- **QUEUE STALL** cannot fire — `#650 Test & Deploy`, queued 2026-08-19T04:04:09Z, is the
  known baseline (item 16); 0 runners online, so nothing to pair it with.
- **DISK** clean — `50928407` 132/200 = **66.0%** (highest), `51415980` 184/420 = 43.8%,
  `49102182` 89/300 = 29.7%, `47913006` 0.8/700 = 0.1%. Byte-identical to 10:30Z, all far
  under 90%.
- **TELEMETRY** clean — `gpu_temp` 60.0 / 38.0 / 49.0 / 53.0 °C, all non-zero, no dead
  frame. `50928407` still `stopped`/`loading`, the stale pair first noted 2026-09-25
  12:30Z — fourth day, not billing GPU, not actioned.

**Nothing changed on the fleet this hour, so Chris was not notified.**

**One correction to 10:30Z's forecast — the cadence does NOT always cancel.** 10:30Z
concluded "each hourly commit now concurrency-cancels the previous hour's run before it can
reach a verdict" and expected the unfinished count to sit at **2** indefinitely. It reads
**1** this hour: `#1234 Test & Deploy` (triggered by 10:30Z's commit `3d358c4` at 10:32:02Z)
ran **54.9 min** and reached its own verdict — **`failure` at 11:26:58Z**, three minutes
before this check. Same failing step as `#1231`: `deploy` job `success`, `test` job
`failure` at *Run test suite (data integrity + browser)*. So the cancel/verdict outcome is
**a race, not a rule** — it turns on whether the hourly commit lands before or after the
previous run's ~55-minute mark, and at a commit time of ~:31 the run clears by ~:27. Expect
the count to alternate between 1 and 2. Test & Deploy has been red for days (#1230, #1231,
#1234 all the same step); hosted `ubuntu-latest`, no Vast cost, outside the five conditions
— still noted, still not actioned, still not new. No new `ml-train` run — newest remains
#554, 2026-09-07.

**Budget — unchanged.** $0.00/h GPU; **$0.0815/h storage = $1.96/day** across the four
stopped boxes. The ≈$41-of-$50 figure remains 03:30Z's **extrapolation, not a reading**.
At $1.96/day with nothing training, idle disk alone reaches the $50 cap around
**2026-10-02**, buying no science. Reported at 08:30Z; per §0e never acted on — no box
stopped, started or destroyed from this session.

**Standing items.** **41 still open, BLOCKING, needs Chris** — the project store is over its
2,000,000-token cap; every project write still fails for every session. ~610 superseded
hourly `fleet-check-*.md` docs are the bulk; deleting project docs is outside this routine's
remit, so **nothing was deleted.** Sent at 08:30Z, unchanged, not re-notified. 15 — 101st
fresh-container bootstrap; re-cloned `earth` (shallow, `3d358c4`) and rewrote `.gh_pat` +
`.vast_key` from `claude/github-access.md` Rule 2b + `claude/vast-access.md` with the
**Write tool, never argv**, `chmod 600`. Routine, not an event. 40, 31, 37 unchanged.
Mapping unchanged (4): `47913006`←`gpu-box-46996216`, `49102182`←`gpu-box-31299601`,
`50928407`←`gpu-box-46694776`, `51415980`←`gpu-box-31947967`.

**CPU-BOUND rule unchanged:** decide against a control's first `stage2_step` `wall_s`
(`#478`, K=144/1024x16/batch256 → `wall_s 240`), **never a threshold**; exclude the k-fold
ridge solves and a run's first 1–2 h of anomaly transform + embed; **name the script**;
write the deadline and threshold down before the evidence closes; price both errors. Never
destroy from this session; never stop a box with a running job.

**For 12:30Z:** nothing in flight, no run to deadline, no box to watch. This commit will
trigger `#1235`; whether 12:30Z sees it `failure` or `cancelled` depends on the race above —
either way it is not a finding. Append here rather than creating a new file.

---

## 2026-09-28 10:30Z — HEALTHY (exit 0), fleet unchanged from 09:30Z. 0 mutations, 0 fleet commits.

`fleet_health.mjs`: `2 run(s) not finished · 0 runner(s) online+idle` / HEALTHY / EXIT=0.
Reading at 10:30:48Z. **Every Vast number byte-identical to 09:30Z** (`fleet_box_detail.mjs`
then a direct `/api/v1/instances/` read).

- **CPU-BOUND** n/a — no box `running`, no job anywhere on the fleet. No threshold used,
  no control read, no script to name.
- **IDLE BURN** clean — 4 instances, all `cur_state=stopped`, 0 of 21 runners online.
  **$0.00/h GPU burn.** No `gpu_box.mjs stop` issued.
- **QUEUE STALL** cannot fire — `#650 Test & Deploy`, queued 2026-08-19T04:04:09Z, is the
  known baseline (item 16); 0 runners online, so nothing to pair it with.
- **DISK** clean — `50928407` 132/200 = **66.0%** (highest), `51415980` 184/420 = 43.8%,
  `49102182` 89/300 = 29.7%, `47913006` 0.8/700 = 0.1%. All far under 90%.
- **TELEMETRY** clean — `gpu_temp` 60.0 / 38.0 / 49.0 / 53.0 °C, all non-zero, no dead
  frame. `50928407` still `stopped`/`loading`, the stale pair first noted 2026-09-25
  12:30Z — fourth day, not billing GPU, not actioned.

**Nothing changed this hour, so Chris was not notified.** The recording series now reads
03:30 · 04:30 · 05:30 · [06:29] · [07:29] · 08:30 · 09:30 · 10:30 — **three consecutive
recorded hours**, so the two-hour hole of 06:29/07:29 looks like a one-off rather than a
continuing fault. 08:30Z's instruction still stands if a third gap appears: chase the
recording path, not the fleet.

**One correction to 09:30Z's forecast, worth a line.** It predicted `#1232 Test & Deploy`
would end **`failure`** shortly, inside the 46.6–58.5 min band. It ended **`cancelled`** at
09:33:11Z — superseded by `#1233`, which the 09:30Z check's own commit `95c4ea0` triggered
at 09:32:54Z. The band was right about the duration (~59 min) and wrong about the verdict,
because the 60-minute check cadence and the ~55-minute Test & Deploy runtime mean **each
hourly commit now concurrency-cancels the previous hour's run before it can fail.** That is
a property of this routine, not of the test suite: the last run to reach its own verdict was
`#1231` (`failure`, 06:32:26Z). So `#650` + `#1233` (in_progress since 09:32:54Z, ~58 min at
this reading) is the two, and the count is expected to stay at 2 indefinitely while the log
keeps committing hourly. All `ubuntu-latest`, no Vast cost. Test & Deploy has been red for
days: still noted, still not actioned, still not new. No new `ml-train` run — newest remains
#554, 2026-09-07.

**Budget — unchanged.** $0.00/h GPU; **$0.0815/h storage = $1.96/day** across the four
stopped boxes (0.01296 + 0.01111 + 0.01852 + 0.03889, re-summed this hour). The ≈$41-of-$50
figure is still 03:30Z's **extrapolation, not a reading** — Vast `/users/current/` exposes no
balance field. At $1.96/day with nothing training, idle disk alone reaches the $50 cap around
**2026-10-02**, buying no science. Per §0e reported, never acted on: no box stopped, started
or destroyed.

**Standing items.** **41 still open, BLOCKING, needs Chris** — the project store is over its
2,000,000-token cap, so every project write still fails for every session; ~610 superseded
hourly `fleet-check-*.md` docs are the bulk, and deleting project docs stays outside this
routine's remit, so **nothing was deleted.** Sent at 08:30Z, unchanged, not re-notified.
15 — 100th fresh-container bootstrap; re-cloned `earth` (shallow, `95c4ea0`) and rewrote
`.gh_pat` + `.vast_key` from `claude/github-access.md` Rule 2b + `claude/vast-access.md`
with the **Write tool, never argv**, `chmod 600`. Routine, not an event.

---

## 2026-09-28 09:30Z — HEALTHY (exit 0), fleet unchanged from 08:30Z. 0 mutations, 0 fleet commits.

`fleet_health.mjs`: `2 run(s) not finished · 0 runner(s) online+idle` / HEALTHY / EXIT=0.

**All five conditions clean; every Vast reading byte-identical to 08:30Z (two frames,
`fleet_box_detail.mjs` then a direct `/api/v1/instances/` read):**

- **CPU-BOUND** n/a — no box `running`, no job anywhere on the fleet. No threshold used,
  no control read, no script to name.
- **IDLE BURN** clean — 4 instances, all `cur_state=stopped`, 0 of 21 runners online.
  **$0.00/h GPU burn.**
- **QUEUE STALL** cannot fire — `#650 Test & Deploy`, queued 2026-08-19T04:04:09Z, is the
  known baseline (item 16); 0 runners online, so nothing to pair it with.
- **DISK** clean — `50928407` 132/200 = **66%** (highest), `51415980` 184/420 = 44%,
  `49102182` 89/300 = 30%, `47913006` 0.8/700 = 0%. All far under 90%.
- **TELEMETRY** clean — all four `gpu_temp` non-zero (60.0 / 38.0 / 49.0 / 53.0 °C).
  No dead frame. `50928407` still `stopped`/`loading`, the stale pair first noted
  2026-09-25 12:30Z. Not billing GPU, not actioned.

**THE HOLE DID NOT RECUR — that is this hour's only finding, and it closes 08:30Z's.**
The scheduled task `trig_018EjzuU6evcGUqM2iBRvRVY` is enabled, cron `29 * * * *`,
`last_fired_at` 2026-09-28T09:29:32Z (this run), next 10:29Z. 08:30Z left its commit
`29d1f7c` and this entry follows it, so the series now reads 03:30 · 04:30 · 05:30 ·
[06:29] · [07:29] · 08:30 · 09:30 — one two-hour gap, not a continuing one. Two consecutive
recorded hours is not yet evidence the recording path is sound; if a third gap appears,
08:30Z's instruction stands — chase the recording path, not the fleet. **Chris not notified
this hour:** no fleet condition, and the two things that would warrant one (item 41, the
budget arithmetic) were both sent at 08:30Z and are unchanged.

**The 1→2 rise in "runs not finished" is the predicted one, not an event.** `#1232 Test &
Deploy`, the 08:30Z check's own commit, has been `in_progress` since 08:34:18Z — ~57 min at
this reading, inside the 46.6–58.5 min / all-`failure` band of the last eight non-cancelled
runs, so expect it to end `failure` shortly. With `#650` that is the two. Test & Deploy has
been red for days: **still noted, still not actioned, still not new.** All `ubuntu-latest`,
no Vast cost. No new `ml-train` run — newest remains #554, 2026-09-07.

**Budget — unchanged, and still the one thing moving.** $0.00/h GPU;
**$0.0815/h storage = $1.96/day** across the four stopped boxes (0.0130 + 0.0111 + 0.0185
+ 0.0389, re-summed this hour). The ≈$41.2-of-$50 figure remains 03:30Z's **extrapolation,
not a reading** — Vast `/users/current/` exposes no balance field. At $1.96/day with nothing
training, idle disk alone reaches the $50 cap around **2026-10-02**, buying no science. Per
§0e reported, never acted on: no box stopped, started or destroyed.

**Standing items.** **41 still open, BLOCKING, needs Chris** — `project_info` read directly
this hour: **knowledge_size 2,000,415 / max 2,000,000**, i.e. over the cap, so every project
write still fails for every session. ~610 superseded hourly `fleet-check-*.md` docs are the
bulk. Deleting project docs stays outside this routine's remit — **nothing was deleted.**
15 — 99th fresh-container bootstrap; re-cloned `earth`, rewrote `.gh_pat` + `.vast_key` from
the project docs with the Write tool, never argv, `chmod 600`. 40, 31, 37 unchanged. Mapping
unchanged (4): `47913006`←`gpu-box-46996216`, `49102182`←`gpu-box-31299601`,
`50928407`←`gpu-box-46694776`, `51415980`←`gpu-box-31947967`.

**CPU-BOUND rule unchanged:** decide against a control's first `stage2_step` `wall_s`
(`#478`, K=144/1024x16/batch256 → `wall_s 240`), **never a threshold**; exclude the k-fold
ridge solves and a run's first 1–2 h of anomaly transform + embed; **name the script**;
write the deadline and threshold down before the evidence closes; price both errors.
Never destroy from this session; never stop a box with a running job.

**For 10:29Z:** nothing in flight, no run to deadline, no box to watch. Expect `#1232` to
have settled `failure` and this commit's own `#1233` to be the unfinished hosted run —
known pattern, not a finding. Append here rather than creating a new file.

---

## 2026-09-28 08:30Z — HEALTHY (exit 0), fleet unchanged from 05:30Z. 0 mutations, 0 fleet commits.

`fleet_health.mjs`: `1 run(s) not finished · 0 runner(s) online+idle` / HEALTHY / EXIT=0.

**All five conditions clean; every Vast reading byte-identical to 05:30Z:**

- **CPU-BOUND** n/a — no box `running`, no job anywhere on the fleet. No threshold used,
  no control read, no script to name.
- **IDLE BURN** clean — 4 instances, all `cur_state=stopped`, 0 of 21 runners online.
  **$0.00/h GPU burn.**
- **QUEUE STALL** cannot fire — `#650 Test & Deploy`, queued 2026-08-19T04:04:09Z, is the
  known baseline (item 16); 0 runners online, so nothing to pair it with.
- **DISK** clean — `50928407` 132/200 = **66%** (highest), `51415980` 184/420 = 44%,
  `49102182` 89/300 = 30%, `47913006` 0.8/700 = 0%. All far under 90%.
- **TELEMETRY** clean — all four `gpu_temp` non-zero (60.0 / 38.0 / 49.0 / 53.0 °C).
  No dead frame. `50928407` still `stopped`/`loading`, the stale pair first noted
  2026-09-25 12:30Z. Not billing GPU, not actioned.

**THE HOURLY SERIES HAS A TWO-HOUR HOLE, AND THAT IS THIS HOUR'S ONLY FINDING.** The
scheduled task is healthy — `trig_018EjzuU6evcGUqM2iBRvRVY`, enabled, cron `29 * * * *`,
`last_fired_at` 2026-09-28T08:30:07Z (this run), next 09:29Z — so **06:29Z and 07:29Z
fired and left NO record anywhere**: no entry in this file, and `main` still stands at
`0cfc1ac`, the 05:30Z check's own commit. Whether those two runs failed early or ran
healthy and never committed is **not knowable from here** — only their absence is
measured. Materially nothing was missed (the fleet was idle and is byte-identical), but
the *mechanism* is the one that produced the ~60-hour gap of 09-25 → 09-28: a check that
leaves no record is indistinguishable from a healthy one from the outside. **Chris
notified this hour** for that reason, with the budget arithmetic below — not for a fleet
condition, of which there is none.

**Hosted lanes — two changes since 05:30Z, both predicted or benign, neither a fleet
condition.** `#1231 Test & Deploy` (the 05:30Z check's own commit) ran 05:33:56Z →
06:32:26Z (58.5 min) and ended `failure` — exactly what the 05:30Z entry said to expect,
inside the 46.6–58.5 min / all-`failure` band of the last eight non-cancelled runs. Test &
Deploy has now been red for days: **still noted, still not actioned, still not new.**
`#58 slatrack fetch 1993-2024` **succeeded** 05:52:05Z → 05:58:16Z — the second
consecutive success after #55/#56 failed on 09-27, so the intermittency the 03:30Z doc
recorded (3 of 14) is currently clearing itself. `#97 GLORYS pull (global)`, `#165 GLORYS
pull`, `#219 tpu-status-mirror` all green. All `ubuntu-latest`, no Vast cost.

**Budget — unchanged, and the one thing still moving.** $0.00/h GPU;
**$0.0815/h storage = $1.96/day** across the four stopped boxes (`storage_total_cost`
summed this hour: 0.0130 + 0.0111 + 0.0185 + 0.0389). The ≈$41.2-of-$50 figure remains
03:30Z's **extrapolation, not a reading** — Vast `/users/current/` exposes no balance
field. At $1.96/day with nothing training, idle disk alone reaches the $50 cap around
**2026-10-02**, buying no science. Per §0e this is reported, never acted on: no box was
stopped, started or destroyed.

**Standing items.** **41 still open, BLOCKING, needs Chris** — re-measured this hour, not
assumed: a deliberately tiny `project_write` (~75 tokens) was refused, *"would exceed the
project's maximum size (~2000000 tokens)"*. Every project write still fails for every
session; ~610 superseded hourly `fleet-check-*.md` docs are the bulk. Deleting project
docs stays outside this routine's remit — **nothing was deleted.** 15 — 98th
fresh-container bootstrap; re-cloned `earth`, rewrote `.gh_pat` + `.vast_key` from the
project docs with the Write tool, never argv, `chmod 600`. 40, 31, 37 unchanged. Mapping
unchanged (4): `47913006`←`gpu-box-46996216`, `49102182`←`gpu-box-31299601`,
`50928407`←`gpu-box-46694776`, `51415980`←`gpu-box-31947967`.

**CPU-BOUND rule unchanged:** decide against a control's first `stage2_step` `wall_s`
(`#478`, K=144/1024x16/batch256 → `wall_s 240`), **never a threshold**; exclude the k-fold
ridge solves and a run's first 1–2 h of anomaly transform + embed; **name the script**;
write the deadline and threshold down before the evidence closes; price both errors.
Never destroy from this session; never stop a box with a running job.

**For 09:29Z:** nothing in flight, no run to deadline, no box to watch. Expect this
commit's own `Test & Deploy` (`#1232`) as an unfinished hosted run, failing in ~50 min —
known pattern, not a finding. **If this file again shows a gap, the recording path itself
is the fault to chase, not the fleet.** Append here rather than creating a new file.

---

## 2026-09-28 05:30Z — HEALTHY (exit 0), unchanged from 04:30Z. 0 mutations, 0 fleet commits.

`fleet_health.mjs`: `1 run(s) not finished · 0 runner(s) online+idle` / HEALTHY / EXIT=0.

**All five conditions clean; every Vast reading is byte-identical to 04:30Z:**

- **CPU-BOUND** n/a — no box `running`, no job anywhere on the fleet. No threshold used,
  no control read, nothing to name.
- **IDLE BURN** clean — 4 instances, all `cur_state=stopped`, 0 runners online+idle.
  **$0.00/h GPU burn.**
- **QUEUE STALL** cannot fire — `#650 Test & Deploy`, queued 2026-08-19T04:04:09Z, is the
  known baseline (item 16); no online+idle runner to pair it with.
- **DISK** clean — `50928407` 132/200 = **66%** (highest), `51415980` 184/420 = 44%,
  `49102182` 89/300 = 30%, `47913006` 0.8/700 = 0%. All far under 90%.
- **TELEMETRY** clean — all four `gpu_temp` non-zero (60.0 / 38.0 / 49.0 / 53.0 °C).
  No dead frame.

`50928407` still reads `cur_state=stopped` / `actual_status=loading` — the same stale pair
first noted 2026-09-25 12:30Z. Not billing GPU, not actioned.

**The 04:30Z entry's one open thread closed by itself, exactly as predicted there.**
`#1229 Test & Deploy` (`36374221421`), whose hosted `test` job had been in step 6 for
~55 min, went `completed/cancelled` at 04:35:06Z — **concurrency-cancelled** by the 04:30Z
check's own commit `e146557`, the same way `#1227` was by `#1228`. Its successor
**`#1230 Test & Deploy`** then ran 04:33:36Z → 05:22:59Z (**49.4 min**) and ended
`failure`, which lands inside the 46.6–57.8 min / all-`failure` band of the last seven
non-cancelled runs of that workflow. So the 2→1 drop in "runs not finished" is that pair
resolving, not a fleet event: hosted `ubuntu-latest` lanes, no Vast cost, outside this
routine's five conditions. **Test & Deploy has now been red for days — still noted, still
not actioned, still not new.** `#97 GLORYS pull (global)` started 05:30:40Z, also hosted;
per the standing note that workflow is reportable on a **failure**, never on spacing.

**Budget unchanged:** $0.00/h GPU, **$0.0815/h storage = $1.96/day** across the four
stopped boxes. The ≈$41.2-of-$50 figure remains 03:30Z's *extrapolation*, not a reading —
Vast `/users/current/` exposes no balance field.

**Standing items:** 41 still open and blocking — every `project_write` still refused, so
this file is again the record; nothing deleted, that stays Chris's call. 15 — 97th
fresh-container bootstrap; re-cloned `earth`, rewrote `.gh_pat` + `.vast_key` from the
project docs with the Write tool, never argv, `chmod 600`. 40, 31, 37 unchanged. Mapping
unchanged (4): `47913006`←`gpu-box-46996216`, `49102182`←`gpu-box-31299601`,
`50928407`←`gpu-box-46694776`, `51415980`←`gpu-box-31947967`.

**CPU-BOUND rule unchanged:** decide against a control's first `stage2_step` `wall_s`
(`#478`, K=144/1024x16/batch256 → `wall_s 240`), **never a threshold**; exclude the k-fold
ridge solves and a run's first 1–2 h of anomaly transform + embed; **name the script**;
write the deadline and threshold down before the evidence closes; price both errors.
Never destroy from this session; never stop a box with a running job.

**For 06:29Z:** nothing in flight, no run to deadline, no box to watch. Expect this
commit's own `Test & Deploy` (`#1231`) to appear as an unfinished hosted run and to fail
in ~50 min — that is the known pattern, not a finding. Append here rather than creating a
new file.

---

## 2026-09-28 04:30Z — HEALTHY (exit 0), unchanged from 03:30Z. 0 mutations, 0 fleet commits.

`fleet_health.mjs`: `2 run(s) not finished · 0 runner(s) online+idle` / HEALTHY / EXIT=0.

**All five conditions clean, and every reading is byte-identical to 03:30Z:**

- **CPU-BOUND** n/a — no box `running`, no job anywhere on the fleet. No threshold
  used, no control read, nothing to name.
- **IDLE BURN** clean — 4 instances, all `cur_state=stopped`, 0 runners online+idle.
  **$0.00/h GPU burn.**
- **QUEUE STALL** cannot fire — `#650 Test & Deploy`, queued 2026-08-19T04:04:09Z, is
  the known baseline (item 16); no online+idle runner to pair it with.
- **DISK** clean — `50928407` 132/200 = **66%** (highest), `51415980` 184/420 = 44%,
  `49102182` 89/300 = 30%, `47913006` 0.8/700 = 0%. All far under 90%.
- **TELEMETRY** clean — all four `gpu_temp` non-zero (60.0 / 38.0 / 49.0 / 53.0 °C).
  No dead frame.

`50928407` still reads `cur_state=stopped` / `actual_status=loading` — the same stale
pair first noted 2026-09-25 12:30Z. Not billing GPU, not actioned.

**The second unfinished run is new but not a fleet condition.** `#1229 Test & Deploy`
(`36374221421`), pushed by the 03:30Z check's own commit `08ce2c6` at 03:34:06Z. Its
`deploy` job succeeded 03:34:51Z; its `test` job is on a **GitHub-hosted
`ubuntu-latest`**, not a fleet box, sitting in step 6 "Run test suite". At the time of
this check that is ~55 min. **That is the normal trajectory for this workflow, not a
hang**: the last six non-cancelled `Test & Deploy` runs ran 46.6–57.8 min and all six
ended `failure` (#1228, #1226, #1223, #1222, #1221, and earlier #1033). So Test &
Deploy has been red for days. Hosted lanes, no Vast cost, outside this routine's five
conditions — **noted, not actioned, and not new this hour.**

**Budget unchanged from 03:30Z:** $0.00/h GPU, **$0.0815/h storage = $1.96/day** across
the four stopped boxes. The ≈$41.2-of-$50 figure remains 03:30Z's *extrapolation*, not a
reading — Vast `/users/current/` exposes no balance field.

**Standing items:** 41 still open and blocking (store reads **2,000,415 / 2,000,000**
this hour — unchanged, every `project_write` still refused; nothing deleted, that stays
Chris's call). 15 — 96th fresh-container bootstrap; re-cloned `earth`, rewrote
`.gh_pat` + `.vast_key` from the project docs with the Write tool, never argv,
`chmod 600`. 40, 31, 37 unchanged. Mapping unchanged (4):
`47913006`←`gpu-box-46996216`, `49102182`←`gpu-box-31299601`,
`50928407`←`gpu-box-46694776`, `51415980`←`gpu-box-31947967`.

**CPU-BOUND rule unchanged:** decide against a control's first `stage2_step` `wall_s`
(`#478`, K=144/1024x16/batch256 → `wall_s 240`), **never a threshold**; exclude the
k-fold ridge solves and a run's first 1–2 h of anomaly transform + embed; **name the
script**; write the deadline and threshold down before the evidence closes; price both
errors. Never destroy from this session; never stop a box with a running job.

**For 05:29Z:** nothing in flight, no run to deadline, no box to watch. Append here
rather than creating a new file.

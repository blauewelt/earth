# Rolling fleet-check log

One compact entry per hourly check. **Deliberately one file, not one file per
hour** — ~610 per-hour `fleet-check-*.md` docs are what pushed the project store
over its 2,000,000-token cap (standing item 41). Do not reintroduce that pattern
here. Long-form records only when an hour actually warrants one; the
2026-09-28 03:30Z doc is the current baseline and stays as its own file.

Newest first.

---

## 2026-09-29 16:30Z — HEALTHY (exit 0, twice), fleet unchanged from 14:30Z. 0 mutations, 0 fleet commits.

`fleet_health.mjs`: `1 run(s) not finished · 0 runner(s) online+idle` / HEALTHY / EXIT=0 at
16:30:3xZ and again at 16:32:57Z. The one unfinished run is `#650` (item 16 baseline).

**Item 42 checked first, per 14:30Z's instruction: the schedule is BACK ON TIME.** This run was
scheduled for 16:29Z and fired at 16:29Z. No separate 15:30Z firing arrived — the 14:30Z entry
below already covers that slot, so skip history gains **15:30Z (09-29, absorbed by the 14:30Z
late fire)** and nothing else. Skip history: 06:30Z/07:30Z, 13:30Z/14:30Z, 22:30Z (all 09-28),
05:30Z, 09:30Z, 14:30Z-late and 15:30Z-absorbed (09-29). Scheduler-side, no fleet consequence —
nothing was in flight to miss. Not notified.

**All five conditions clean; every Vast field byte-identical to 14:30Z (and so back to 04:30Z),
across two frames 114 s apart (16:30:59Z and 16:32:53Z):**

- **CPU-BOUND** n/a — no box `running`, no job anywhere on the fleet, 0 runners online.
  `gpu_util` and `cpu_util` 0 on all four. No threshold used, no control read, no script to
  name, no deadline to write down.
- **IDLE BURN** clean — 4 instances, all `cur_state=stopped` AND `intended_status=stopped`.
  **$0.00/h GPU burn.** No `gpu_box.mjs stop` issued. §0e not needed — nothing dispatched.
- **QUEUE STALL** cannot fire — `in_progress` is **empty (0 runs)**; `queued` is exactly
  `#650 Test & Deploy`, queued 2026-08-19T04:04:09Z (item 16 baseline). 0 runners online,
  nothing to pair it with.
- **DISK** clean — `50928407` 132/200 = **66.0%** (highest), `51415980` 184/420 = 43.8%,
  `49102182` 89/300 = 29.7%, `47913006` 0.8/700 = 0.1%. All far under 90%.
- **TELEMETRY** clean — `gpu_temp` 60.0 / 38.0 / 49.0 / 53.0 °C in both frames, `vmem`
  0.473 / 0.471 / 0.340 / 0.369. No dead frame. `50928407` still `stopped`/`loading`, the stale
  pair first noted 2026-09-25 12:30Z — eighteenth day, not billing GPU, not actioned.

**14:30Z's prediction landed; the failure band's lower edge moved again.** `#1257 Test & Deploy`
— spawned by the 14:30Z check's own commit `be91031` — ran 15:38:05Z → 16:14:44Z and ended
`failure` in **36.6 min**, below the 42.3–58.0 min band. **Ten consecutive `Test & Deploy`
failures now span 36.6–58.0 min.** Red for days; noted, not actioned, not new — only the band's
lower edge moved, for the second check running. One other hosted lane closed: **`#225
tpu-status-mirror` `success`** 15:47:31Z → 15:47:52Z (21 s), the routine mirror whose previous
run was `#224` at 09:02Z — green, expected, `ubuntu-latest`, no fleet box, no Vast cost. `#62`
slatrack did not reappear (not due before ~18:30Z); `#72`, `#169`, `#101`, `#13` did not
reappear, as predicted. Runner registrations **19, unchanged**, all offline, 0 busy.

**Budget unchanged:** $0.00/h GPU; **$0.0815/h storage = $1.96/day** across the four stopped boxes
(`storage_total_cost` 0.012963 + 0.011111 + 0.018519 + 0.038889 — identical to 14:30Z and every
hour back to 00:30Z). The ≈$41-of-$50 figure is still 2026-09-28 03:30Z's **extrapolation, not a
reading** — Vast `/users/current/` exposes no balance field.

**Item 43 (open, dated):** idle-storage-reaches-$50 projection ≈2026-10-02; the re-notify is owed
on **2026-10-01** and not before — today is 09-29, so **not re-notified this hour**, deliberately.
`47913006` remains **stop-only, never destroy** (sole copy of the 213 GB family-5 daily tensor).

**Standing items.** **41 still open, BLOCKING** — `project_info` WAS called this hour (the
container was fresh and the project docs were the credential route), and the store reads
**2,000,415 / 2,000,000 tokens, i.e. still over cap**. The newest project doc is still
`claude/fleet-check-2026-09-25-1530Z.md`: no project write has landed in nine days. No project
write was attempted. This file remains the record; nothing deleted, which stays Chris's call.
42 open, **did not fire** (on time again). 40, 37, 31, 16 unchanged. 15 — **126th** fresh-container
bootstrap; the container started with no `earth`, no `.gh_pat` and no `.vast_key`. Re-cloned at
`be91031` (`--depth 1 --filter=blob:none`, first try) and rewrote both credential files from the
project docs with the **Write tool, never argv**, `chmod 600`. Routine, not an event. Mapping
unchanged (4): `47913006`←`gpu-box-46996216`, `49102182`←`gpu-box-31299601`,
`50928407`←`gpu-box-46694776`, `51415980`←`gpu-box-31947967`.

**`ml/OVERVIEW.md`'s stamp reads 2026-09-16 ~20:30Z** — thirteen days stale, which is correct
rather than alarming: nothing has been dispatched or harvested since, so §0g's "sessions that
only monitor update the stamp only if they changed something" applies. Read before deciding
anything was anomalous, as the routine requires; it names nothing in flight.

**The telemetry note holds and was used:** the per-condition telemetry above is a direct Node
`fetch` of `GET /api/v1/instances/` (**v1, not v0**; v0 returns an empty `instances` array with
this key, which looks exactly like "no boxes exist").

**CPU-BOUND rule unchanged:** decide against a control's first `stage2_step` `wall_s` (`#478`,
K=144/1024x16/batch256 → `wall_s 240`), **never a threshold**; exclude the k-fold ridge solves and
a run's first 1–2 h of anomaly transform + embed; **name the script**; write the deadline and
threshold down before the evidence closes; price both errors. Never destroy from this session;
never stop a box with a running job.

**For the next check:** nothing in flight on the fleet, no run to deadline, no box to watch.
Expect this commit's own `Test & Deploy` as a new unfinished hosted run, failing in **36.6–58
min** — known pattern, not a finding. `#63 slatrack` becomes due ~18:30Z; the slatrack escalation
threshold set at 13:30Z stands (report only if the *same* lane fails twice consecutively, or one
run fails *all six* lanes). **Check item 42 first**: if the entry directly below is not this one,
or the fire time again trails the scheduled time by more than a few minutes, the schedule slipped
again. Append here rather than creating a new file.

**No notification sent** — fleet healthy, every Vast field unchanged, the schedule recovered, and
the hour's two closed hosted runs were both predicted (`#1257` in its known failure class,
`#225` a routine green mirror). See item 43 for the one dated thing that warrants a ping on
10-01.

---

## 2026-09-29 14:30Z (fired 15:33Z) — HEALTHY (exit 0, twice), fleet unchanged from 13:30Z. 0 mutations, 0 fleet commits.

`fleet_health.mjs`: `1 run(s) not finished · 0 runner(s) online+idle` / HEALTHY / EXIT=0 at
15:34:5xZ and again at 15:37:3xZ. The one unfinished run is `#650` (item 16 baseline).

**Item 42 checked first, per 13:30Z's instruction: it DID recur, in a new form.** This run was
scheduled for **14:29:00Z** and fired at **15:33:26Z** — a **64-minute delay**, not a clean skip:
the slot was delivered late rather than dropped, and no separate 15:29Z firing arrived. Net effect
on the record is the same as a skip — there is no 14:30Z-on-time entry and none for 15:30Z; this
entry covers both. Three consecutive on-time hours (11:30Z–13:30Z) ended here. Skip history now:
06:30Z/07:30Z, 13:30Z/14:30Z, 22:30Z (all 09-28), 05:30Z, 09:30Z and **14:30Z-late (09-29)**.
Scheduler-side, no fleet consequence — nothing was in flight to miss. Not notified.

**All five conditions clean; every Vast field byte-identical to 13:30Z (and so back to 04:30Z),
across two frames 137 s apart (15:35:02Z and 15:37:20Z):**

- **CPU-BOUND** n/a — no box `running`, no job anywhere on the fleet, 0 runners online.
  `gpu_util` and `cpu_util` 0 on all four. No threshold used, no control read, no script to
  name, no deadline to write down.
- **IDLE BURN** clean — 4 instances, all `cur_state=stopped` AND `intended_status=stopped`.
  **$0.00/h GPU burn.** No `gpu_box.mjs stop` issued. §0e not needed — nothing dispatched.
- **QUEUE STALL** cannot fire — `in_progress` is **empty (0 runs)**; `queued` is exactly
  `#650 Test & Deploy`, queued 2026-08-19T04:04:09Z (item 16 baseline). 0 runners online,
  nothing to pair it with.
- **DISK** clean — `50928407` 132/200 = **66.0%** (highest), `51415980` 184/420 = 43.8%,
  `49102182` 89/300 = 29.7%, `47913006` 0.8/700 = 0.1%. All far under 90%.
- **TELEMETRY** clean — `gpu_temp` 60.0 / 38.0 / 49.0 / 53.0 °C in both frames, `vmem`
  0.473 / 0.471 / 0.340 / 0.369. No dead frame. `50928407` still `stopped`/`loading`, the stale
  pair first noted 2026-09-25 12:30Z — eighteenth day, not billing GPU, not actioned.

**13:30Z's predictions landed exactly.** `#1256 Test & Deploy` — spawned by the 13:30Z check's own
commit `4edc755` — ran 13:37:01Z → 14:33:24Z and ended `failure` in **56.4 min**, inside the
42–58 min band. Nine consecutive `Test & Deploy` failures now span **42.3–58.0 min**. Red for
days; noted, not actioned, not new. `#72`, `#169`, `#101`, `#13` did not reappear, as predicted;
`#63 slatrack` is not due before ~18:30Z and did not appear. **No hosted run closed this hour
other than `#1256`** — the quietest hour in this series. Runner registrations **19, unchanged**,
all offline, 0 busy.

**Budget unchanged:** $0.00/h GPU; **$0.0815/h storage = $1.96/day** across the four stopped boxes
(`storage_total_cost` 0.012963 + 0.011111 + 0.018519 + 0.038889 — identical to 13:30Z and every
hour back to 00:30Z). The ≈$41-of-$50 figure is still 2026-09-28 03:30Z's **extrapolation, not a
reading** — Vast `/users/current/` exposes no balance field.

**Item 43 (open, dated):** idle-storage-reaches-$50 projection ≈2026-10-02; the re-notify is owed
on **2026-10-01** and not before — today is 09-29, so **not re-notified this hour**, deliberately.
`47913006` remains **stop-only, never destroy** (sole copy of the 213 GB family-5 daily tensor).

**Standing items.** **41 still open, BLOCKING** — `project_info` was NOT called this hour, so no
fresh token count was taken; `project_read` answered normally for `claude/vast-access.md` and
`claude/github-access.md` (reads are unaffected) and the newest project doc is still
`claude/fleet-check-2026-09-25-1530Z.md`, i.e. no project write has landed in five days. No
project write was attempted. This file remains the record; nothing deleted, which stays Chris's
call. 42 open, **fired** (see above). 40, 37, 31, 16 unchanged. 15 — **125th** fresh-container
bootstrap; the container started with no `earth`, no `.gh_pat` and no `.vast_key`. Re-cloned at
`4edc755` (`--depth 1`, first try) and rewrote both credential files from the project docs with
the **Write tool, never argv**, `chmod 600`. Routine, not an event. Mapping unchanged (4):
`47913006`←`gpu-box-46996216`, `49102182`←`gpu-box-31299601`, `50928407`←`gpu-box-46694776`,
`51415980`←`gpu-box-31947967`.

**The telemetry note holds and was used:** the per-condition telemetry above is a direct Node
`fetch` of `GET /api/v1/instances/` (**v1, not v0**; v0 returns an empty `instances` array with
this key, which looks exactly like "no boxes exist").

**CPU-BOUND rule unchanged:** decide against a control's first `stage2_step` `wall_s` (`#478`,
K=144/1024x16/batch256 → `wall_s 240`), **never a threshold**; exclude the k-fold ridge solves and
a run's first 1–2 h of anomaly transform + embed; **name the script**; write the deadline and
threshold down before the evidence closes; price both errors. Never destroy from this session;
never stop a box with a running job.

**For the next check:** nothing in flight on the fleet, no run to deadline, no box to watch.
Expect this commit's own `Test & Deploy` as a new unfinished hosted run, failing in **42–58 min**
— known pattern, not a finding. `#63 slatrack` becomes due ~18:30Z; the slatrack escalation
threshold set at 13:30Z stands (report only if the *same* lane fails twice consecutively, or one
run fails *all six* lanes). **Check item 42 first**: if the entry directly below is not this one,
or the fire time again trails the scheduled time by more than a few minutes, the schedule slipped
again. Append here rather than creating a new file.

**No notification sent** — fleet healthy, every Vast field unchanged, and the hour's one closed
hosted run (`#1256`, inside its known failure band) was predicted. The 64-minute schedule delay is
item 42, scheduler-side, with nothing in flight to miss. See item 43 for the one dated thing that
warrants a ping on 10-01.

---

## 2026-09-29 13:30Z — HEALTHY (exit 0, twice), fleet unchanged from 12:30Z. 0 mutations, 0 fleet commits.

`fleet_health.mjs`: `1 run(s) not finished · 0 runner(s) online+idle` / HEALTHY / EXIT=0 at
13:33:16Z and again at 13:35:2xZ. Both frames identical — the one unfinished run is `#650`
(item 16 baseline), not anything that started this hour.

**Item 42 checked first, per 12:30Z's instruction: it did NOT recur** — the entry directly below
is 12:30Z, so the schedule has now held three consecutive hours. Skip history unchanged: 06:30Z/
07:30Z, 13:30Z/14:30Z, 22:30Z (all 09-28), 05:30Z and 09:30Z (09-29).

**All five conditions clean; every Vast field byte-identical to 12:30Z (and so back to 04:30Z),
across two frames 121 s apart (13:33:31Z and 13:35:32Z):**

- **CPU-BOUND** n/a — no box `running`, no job anywhere on the fleet, 0 runners online.
  `gpu_util` and `cpu_util` 0 on all four. No threshold used, no control read, no script to
  name, no deadline to write down.
- **IDLE BURN** clean — 4 instances, all `cur_state=stopped` AND `intended_status=stopped`.
  **$0.00/h GPU burn.** No `gpu_box.mjs stop` issued.
- **QUEUE STALL** cannot fire — `in_progress` is **empty (0 runs)**; `queued` is exactly
  `#650 Test & Deploy`, queued 2026-08-19T04:04:09Z (item 16 baseline). 0 runners online,
  nothing to pair it with.
- **DISK** clean — `50928407` 132/200 = **66.0%** (highest), `51415980` 184/420 = 43.8%,
  `49102182` 89/300 = 29.7%, `47913006` 0.8/700 = 0.1%. All far under 90%.
- **TELEMETRY** clean — `gpu_temp` 60.0 / 38.0 / 49.0 / 53.0 °C in both frames, `vmem`
  0.473 / 0.471 / 0.340 / 0.369. No dead frame. `50928407` still `stopped`/`loading`, the stale
  pair first noted 2026-09-25 12:30Z — seventeenth day, not billing GPU, not actioned.

**12:30Z's predictions landed, with one band correction.** `#1255 Test & Deploy` — spawned by the
12:30Z check's own commit `25ed6c9` — ran 12:35:46Z → 13:18:04Z and ended `failure`, but in
**42.3 min, below the 45–58 min band**. Eight consecutive `Test & Deploy` failures now span
**42.3–58.0 min** (#1247 45.5, #1249 56.9, #1250 51.7, #1251 56.3, #1252 56.8, #1253 56.1,
#1254 55.2, #1255 42.3). Red for days; still noted, still not actioned, still not new — only the
band's lower edge moved. `#72`, `#169` did not reappear, as predicted.

**Three hosted lanes closed this hour; two green, one is the known slatrack partial failure.**
`#13 Daily loitering refresh` `success` 12:45:01Z (15 s). `#101 GLORYS pull (global)` `success`
13:00:13Z → 13:02:38Z (2.4 min). **`#62 slatrack fetch 1993-2024` `failure`** 13:19:01Z →
13:23:17Z (4.3 min) — **one lane of six**, `fetch (1993-1999)`, failing at step 8 *"Every year of
the lane must carry a done.json"*; the other five lanes green. This is the same intermittent
hosted pattern already logged at `#59` (09-28 14:33Z, lane `2019-2024`), `#56`, `#55` and `#48`:
**5 failures in the 15 runs since 09-25, a different lane each time, always `ubuntu-latest`, no
fleet box and no Vast cost.** Logged, not actioned, not notified. **Escalation threshold, new
this hour:** report only if the *same* lane fails twice consecutively, or a run fails *all six*
lanes — either would make it a lane-specific defect rather than the flaky-fetch pattern it
currently is. Runner registrations **19, unchanged**, all offline, 0 busy.

**Budget unchanged:** $0.00/h GPU; **$0.0815/h storage = $1.96/day** across the four stopped boxes
(`storage_total_cost` 0.012963 + 0.011111 + 0.018519 + 0.038889 — identical to 12:30Z and every
hour back to 00:30Z). The ≈$41-of-$50 figure is still 2026-09-28 03:30Z's **extrapolation, not a
reading**.

**Item 43 (open, dated):** idle-storage-reaches-$50 projection ≈2026-10-02; the re-notify is owed
on **2026-10-01** and not before — today is 09-29, so **not re-notified this hour**, deliberately.
`47913006` remains **stop-only, never destroy** (sole copy of the 213 GB family-5 daily tensor).

**Standing items.** **41 still open, BLOCKING** — `project_info` was NOT called this hour, so no
fresh token count was taken; `project_read` answered normally for both credential docs (reads are
unaffected) and the newest project doc is still `claude/fleet-check-2026-09-25-1530Z.md`, i.e. no
project write has landed in five days. No project write was attempted. This file remains the
record; nothing deleted, which stays Chris's call. 42 open, did not fire. 40, 37, 31, 16 unchanged.
15 — **124th** fresh-container bootstrap; the container started with no `earth`, no `.gh_pat` and
no `.vast_key`, and the ambient `GH_TOKEN` again returned **401 Bad credentials** (it is not an
API PAT — do not retry it), so the project docs were once more the only route. Re-cloned at
`25ed6c9` (`--depth 1`, first try) and rewrote both credential files with the **Write tool, never
argv**, `chmod 600`. Routine, not an event. Mapping unchanged (4): `47913006`←`gpu-box-46996216`,
`49102182`←`gpu-box-31299601`, `50928407`←`gpu-box-46694776`, `51415980`←`gpu-box-31947967`.

**The telemetry note holds and was used:** the per-condition telemetry above is a direct Node
`fetch` of `GET /api/v1/instances/` (**v1, not v0**; v0 returns an empty `instances` array with
this key, which looks exactly like "no boxes exist").

**CPU-BOUND rule unchanged:** decide against a control's first `stage2_step` `wall_s` (`#478`,
K=144/1024x16/batch256 → `wall_s 240`), **never a threshold**; exclude the k-fold ridge solves and
a run's first 1–2 h of anomaly transform + embed; **name the script**; write the deadline and
threshold down before the evidence closes; price both errors. Never destroy from this session;
never stop a box with a running job.

**For 14:30Z:** nothing in flight on the fleet, no run to deadline, no box to watch. Expect this
commit's own `Test & Deploy` as a new unfinished hosted run, failing in **42–58 min** — known
pattern, not a finding. `#72 Daily forecast refresh`, `#169 GLORYS pull`, `#101 GLORYS pull
(global)` and `#13 Daily loitering refresh` have all run for today and should not reappear;
`slatrack` fires every ~5–8 h, so `#63` is not due before ~18:30Z. **Check item 42 first**: if the
entry directly below is not 13:30Z, the schedule skipped again. Append here rather than creating a
new file.

**No notification sent** — healthy, unchanged, and the hour's three closed hosted runs (`#1255`
inside its known failure pattern at a new lower band edge, `#13`/`#101` green on schedule,
`#62 slatrack` one lane of six on the already-logged intermittent pattern) are all expected. See
item 43 for the one dated thing that warrants a ping on 10-01.

---

## 2026-09-29 12:30Z — HEALTHY (exit 0, twice), fleet unchanged from 11:30Z. 0 mutations, 0 fleet commits.

`fleet_health.mjs`: `1 run(s) not finished · 0 runner(s) online+idle` / HEALTHY / EXIT=0 at
12:31:5xZ and again at 12:35:2xZ. Both frames identical — no daily-refresh run straddling the
check this hour, so the count did not move.

**Item 42 checked first, per 11:30Z's instruction: it did NOT recur** — the entry directly below
is 11:30Z, so the schedule held for a second consecutive hour. Skip history unchanged: 06:30Z/
07:30Z, 13:30Z/14:30Z, 22:30Z (all 09-28), 05:30Z and 09:30Z (09-29).

**All five conditions clean; every Vast field byte-identical to 11:30Z (and so back to 04:30Z),
across two frames 89 s apart (12:32:44Z and 12:34:13Z):**

- **CPU-BOUND** n/a — no box `running`, no job anywhere on the fleet, 0 runners online.
  `gpu_util` and `cpu_util` 0 on all four. No threshold used, no control read, no script to
  name, no deadline to write down.
- **IDLE BURN** clean — 4 instances, all `cur_state=stopped` AND `intended_status=stopped`.
  **$0.00/h GPU burn.** No `gpu_box.mjs stop` issued.
- **QUEUE STALL** cannot fire — `in_progress` is **empty (0 runs)**; `queued` is exactly
  `#650 Test & Deploy`, queued 2026-08-19T04:04:09Z (item 16 baseline). 0 runners online,
  nothing to pair it with.
- **DISK** clean — `50928407` 132/200 = **66.0%** (highest), `51415980` 184/420 = 43.8%,
  `49102182` 89/300 = 29.7%, `47913006` 0.75/700 = 0.1%. All far under 90%.
- **TELEMETRY** clean — `gpu_temp` 60.0 / 38.0 / 49.0 / 53.0 °C in both frames, `vmem`
  0.473 / 0.471 / 0.340 / 0.369. No dead frame. `50928407` still `stopped`/`loading`, the stale
  pair first noted 2026-09-25 12:30Z — sixteenth day, not billing GPU, not actioned.

**11:30Z's two predictions both landed, so neither is a finding.** `#1254 Test & Deploy` —
spawned by the 11:30Z check's own commit `354237b` — ran 11:34:34Z → 12:29:43Z (**55.2 min**) and
ended `failure`. **Inside the 45–58 min band, now unbroken across seven: #1247 (45.5 m), #1249
(56.9 m), #1250 (51.7 m), #1251 (56.3 m), #1252 (56.8 m), #1253 (56.1 m), #1254 (55.2 m), all
`failure`.** Red for days; still noted, still not actioned, still not new. And `#72 Daily forecast
refresh` did NOT reappear, as 11:30Z said it should not — it had already run for today (green,
2.3 min).

**One hosted lane closed this hour, and it is another ordinary scheduled one.** `#169 GLORYS pull`
(`schedule`, `ubuntu-latest`) ran 12:16:44Z → 12:18:33Z, **`success` in 1.8 min**. No fleet box, no
Vast cost. Runner registrations **19, unchanged**, all offline, 0 busy.

**Budget unchanged:** $0.00/h GPU; **$0.0815/h storage = $1.96/day** across the four stopped boxes
(`storage_total_cost` 0.012963 + 0.011111 + 0.018519 + 0.038889 — identical to 11:30Z and every
hour back to 00:30Z). The ≈$41-of-$50 figure is still 2026-09-28 03:30Z's **extrapolation, not a
reading**.

**Item 43 (open, dated):** idle-storage-reaches-$50 projection ≈2026-10-02; the re-notify is owed
on **2026-10-01** and not before — today is 09-29, so **not re-notified this hour**, deliberately.
`47913006` remains **stop-only, never destroy** (sole copy of the 213 GB family-5 daily tensor).

**Standing items.** **41 still open, BLOCKING** — `project_info` was NOT called this hour, so no
fresh token count was taken; `project_read` and `project_search` both answered normally (reads are
unaffected) and the newest project doc is still `claude/fleet-check-2026-09-25-1530Z.md`, i.e. no
project write has landed in four days. No project write was attempted. This file remains the
record; nothing deleted, which stays Chris's call. 42 open, did not fire. 40, 37, 31, 16 unchanged.
15 — **123rd** fresh-container bootstrap; the container started with no `earth`, no `.gh_pat` and
no `.vast_key`, so the project docs were again the only route to the credentials. Re-cloned at
`354237b` (`--depth 1`, first try) and rewrote both credential files with the **Write tool, never
argv**, `chmod 600`. Routine, not an event. Mapping unchanged (4): `47913006`←`gpu-box-46996216`,
`49102182`←`gpu-box-31299601`, `50928407`←`gpu-box-46694776`, `51415980`←`gpu-box-31947967`.

**The telemetry note holds and was used:** the per-condition telemetry above is a direct Node
`fetch` of `GET /api/v1/instances/` (**v1, not v0**; v0 returns an empty `instances` array with
this key, which looks exactly like "no boxes exist").

**CPU-BOUND rule unchanged:** decide against a control's first `stage2_step` `wall_s` (`#478`,
K=144/1024x16/batch256 → `wall_s 240`), **never a threshold**; exclude the k-fold ridge solves and
a run's first 1–2 h of anomaly transform + embed; **name the script**; write the deadline and
threshold down before the evidence closes; price both errors. Never destroy from this session;
never stop a box with a running job.

**For 13:30Z:** nothing in flight on the fleet, no run to deadline, no box to watch. Expect this
commit's own `Test & Deploy` as a new unfinished hosted run, failing in 45–58 min — known pattern,
not a finding. `#72 Daily forecast refresh` and `#169 GLORYS pull` have both already run for today
and should not reappear. **Check item 42 first**: if the entry directly below is not 12:30Z, the
schedule skipped again. Append here rather than creating a new file.

**No notification sent** — healthy, unchanged, and the hour's two closed runs (`#1254` inside its
known band, `#169` green on its normal schedule) are both expected. See item 43 for the one dated
thing that warrants a ping on 10-01.

---

## 2026-09-29 11:30Z — HEALTHY (exit 0, twice), fleet unchanged from 10:30Z. 0 mutations, 0 fleet commits.

`fleet_health.mjs`: `2 run(s) not finished · 0 runner(s) online+idle` / HEALTHY / EXIT=0 at
11:30:5xZ, and `1 run(s) not finished` / HEALTHY / EXIT=0 at 11:34:1xZ — the count fell by one
because `#72 Daily forecast refresh` finished between the frames, not because anything changed on
the fleet.

**Item 42 checked first, per 10:30Z's instruction: it did NOT recur** — the entry directly below
is 10:30Z, so the schedule held. Skip history unchanged: 06:30Z/07:30Z, 13:30Z/14:30Z, 22:30Z
(all 09-28), 05:30Z and 09:30Z (09-29).

**All five conditions clean; every Vast field byte-identical to 10:30Z (and so back to 04:30Z),
across two frames 94 s apart (11:31:35Z and 11:33:09Z):**

- **CPU-BOUND** n/a — no box `running`, no job anywhere on the fleet, 0 runners online.
  `gpu_util` and `cpu_util` 0 on all four. No threshold used, no control read, no script to
  name, no deadline to write down.
- **IDLE BURN** clean — 4 instances, all `cur_state=stopped` AND `intended_status=stopped`.
  **$0.00/h GPU burn.** No `gpu_box.mjs stop` issued.
- **QUEUE STALL** cannot fire — `queued` is exactly `#650 Test & Deploy`, queued
  2026-08-19T04:04:09Z (item 16 baseline); 0 runners online, nothing to pair it with. The one
  `in_progress` run at the first frame was `#72`, on `ubuntu-latest`, not a fleet box.
- **DISK** clean — `50928407` 132/200 = **66.0%** (highest), `51415980` 184/420 = 43.8%,
  `49102182` 89/300 = 29.7%, `47913006` 0.75/700 = 0.1%. All far under 90%.
- **TELEMETRY** clean — `gpu_temp` 60.0 / 38.0 / 49.0 / 53.0 °C in both frames, `vmem`
  0.473 / 0.471 / 0.340 / 0.369. No dead frame. `50928407` still `stopped`/`loading`, the stale
  pair first noted 2026-09-25 12:30Z — fifteenth day, not billing GPU, not actioned.

**10:30Z's prediction landed exactly, so it is not a finding.** `#1253 Test & Deploy` — spawned by
the 10:30Z check's own commit `cb7ccb8` — ran 10:34:17Z → 11:30:22Z (**56.1 min**) and ended
`failure`. **Inside the 45–58 min band, now unbroken across #1247 (45.5 m), #1249 (56.9 m), #1250
(51.7 m), #1251 (56.3 m), #1252 (56.8 m) and #1253 (56.1 m), all `failure`.** Test & Deploy has
been red for days; still noted, still not actioned, still not new.

**One new hosted lane this hour, and it is the ordinary daily one.** `#72 Daily forecast refresh`
(`.github/workflows/refresh-forecast.yml`, `schedule`, `ubuntu-latest`, job `refresh` on a hosted
runner) started 11:29:59Z — 90 s before this check — and finished during it. Its last five runs
(#67–#71, 09-24 → 09-28) are all `success` in 2–3 min, so a run of it in flight at 11:30Z is the
schedule working, **not an anomaly**; it is why the first frame read 2 unfinished runs and the
second read 1. No fleet box, no Vast cost. Runner registrations **19, unchanged**, all offline.

**Budget unchanged:** $0.00/h GPU; **$0.0815/h storage = $1.96/day** across the four stopped boxes
(`storage_total_cost` 0.012963 + 0.011111 + 0.018519 + 0.038889 — identical to 10:30Z and every
hour back to 00:30Z). The ≈$41-of-$50 figure is still 2026-09-28 03:30Z's **extrapolation, not a
reading**.

**Item 43 (open, dated):** idle-storage-reaches-$50 projection ≈2026-10-02; the re-notify is owed
on **2026-10-01** and not before — today is 09-29, so **not re-notified this hour**, deliberately.
`47913006` remains **stop-only, never destroy** (sole copy of the 213 GB family-5 daily tensor).

**Standing items.** **41 still open, BLOCKING** — `project_info` WAS called this hour (this
container started with no repo at all, so the project docs were the only way to recover
`github-access.md` Rule 2b and `vast-access.md`) and it reads **660 docs, `knowledge_size`
2,000,415 / `max_knowledge_size` 2,000,000 — the store is OVER its cap**, with
`claude/fleet-check-2026-09-25-1530Z.md` still the newest doc, i.e. no project write has landed in
four days. This file remains the record; nothing deleted, which stays Chris's call. 42 open, did
not fire. 40, 37, 31, 16 unchanged. 15 — **122nd** fresh-container bootstrap; no `earth`, no
`.gh_pat`, no `.vast_key`. Re-cloned at `cb7ccb8` (`--depth 50`, first try) and rewrote both
credential files with the **Write tool, never argv**, `chmod 600`. Routine, not an event. Mapping
unchanged (4): `47913006`←`gpu-box-46996216`, `49102182`←`gpu-box-31299601`,
`50928407`←`gpu-box-46694776`, `51415980`←`gpu-box-31947967`.

**The telemetry note holds and was used:** the per-condition telemetry above is a direct Node
`fetch` of `GET /api/v1/instances/` (**v1, not v0**; v0 returns an empty `instances` array with
this key, which looks exactly like "no boxes exist").

**CPU-BOUND rule unchanged:** decide against a control's first `stage2_step` `wall_s` (`#478`,
K=144/1024x16/batch256 → `wall_s 240`), **never a threshold**; exclude the k-fold ridge solves and
a run's first 1–2 h of anomaly transform + embed; **name the script**; write the deadline and
threshold down before the evidence closes; price both errors. Never destroy from this session;
never stop a box with a running job.

**For 12:30Z:** nothing in flight on the fleet, no run to deadline, no box to watch. Expect this
commit's own `Test & Deploy` as a new unfinished hosted run, failing in 45–58 min — known pattern,
not a finding. `#72 Daily forecast refresh` has already run for today and should not reappear.
**Check item 42 first**: if the entry directly below is not 11:30Z, the schedule skipped again.
Append here rather than creating a new file.

**No notification sent** — healthy, unchanged, the hour's two closed runs (`#1253` inside its known
band, `#72` green on its normal daily schedule) are both expected. See item 43 for the one dated
thing that warrants a ping on 10-01.

---

## 2026-09-29 10:30Z — HEALTHY (exit 0, twice), fleet unchanged from 08:30Z. 0 mutations, 0 fleet commits.

`fleet_health.mjs`: `1 run(s) not finished · 0 runner(s) online+idle` / HEALTHY / EXIT=0 at
10:31:0xZ and again at 10:35:1xZ.

**Item 42 checked first, per 08:30Z's instruction: it DID recur — the entry directly below is
08:30Z, not 09:30Z, so the 09:30Z check never ran.** Skip history now 06:30Z/07:30Z, 13:30Z/14:30Z,
22:30Z (all 09-28), 05:30Z and 09:30Z (09-29). **Not notified:** the fleet was idle across the gap
— no box `running`, no job, $0/h GPU — and 08:30Z's own rule is that a skip across an idle fleet is
not a notify, a skip with a box `running` is. Nothing was unwatched that needed watching.

**All five conditions clean; every Vast field byte-identical to 08:30Z (and so to 07:30Z, 06:30Z,
04:30Z), across two frames ~50 s apart (10:31:0xZ and 10:34:0xZ):**

- **CPU-BOUND** n/a — no box `running`, no job anywhere on the fleet, 0 runners online.
  `gpu_util` and `cpu_util` 0 on all four. No threshold used, no control read, no script to
  name, no deadline to write down.
- **IDLE BURN** clean — 4 instances, all `cur_state=stopped` AND `intended_status=stopped`.
  **$0.00/h GPU burn.** No `gpu_box.mjs stop` issued.
- **QUEUE STALL** cannot fire — `in_progress` is **empty (0 runs)**; `queued` is exactly
  `#650 Test & Deploy`, queued 2026-08-19T04:04:09Z (item 16 baseline). 0 runners online,
  nothing to pair it with. The single "run not finished" is `#650` and nothing else.
- **DISK** clean — `50928407` 132/200 = **66.0%** (highest), `51415980` 184/420 = 43.8%,
  `49102182` 89/300 = 29.7%, `47913006` 0.8/700 = 0.1%. All far under 90%.
- **TELEMETRY** clean — `gpu_temp` 60.0 / 38.0 / 49.0 / 53.0 °C in both frames, `vmem`
  0.473 / 0.471 / 0.340 / 0.369. No dead frame. `50928407` still `stopped`/`loading`, the
  stale pair first noted 2026-09-25 12:30Z — fourteenth day, not billing GPU, not actioned.

**08:30Z's one prediction landed exactly as predicted, so it is not a finding.** `#1252 Test &
Deploy` — spawned by the 08:30Z check's own commit `19b5046` — ran 08:35:03Z → 09:31:50Z
(**56.8 min**) and ended `failure`. **Inside the 45–58 min band, now unbroken across #1247
(45.5 m), #1249 (56.9 m), #1250 (51.7 m), #1251 (56.3 m) and #1252 (56.8 m), all `failure`.**
Test & Deploy has been red for days; **still noted, still not actioned, still not new.** It
finished ~1 h before this check, so again there is **no unfinished hosted run at all** — which is
why the count read 1 and not 2. Other hosted lanes unchanged and green: `#224 tpu-status-mirror`
09:02:21Z→09:02:39Z success is the only new run since 08:30Z, and it is green; `#61 slatrack fetch`
06:11Z, `#100 GLORYS pull (global)` 05:49Z all still success. All `ubuntu-latest`, **no fleet box,
no Vast cost. Runner registrations 19, unchanged** (21 on 09-25 → 20 → 19), all `offline
busy=false`. Nothing is meant to be in flight, and nothing is.

**Budget unchanged:** $0.00/h GPU; **$0.0815/h storage = $1.96/day** across the four stopped
boxes (`storage_total_cost` 0.012963 + 0.011111 + 0.018519 + 0.038889 — identical to 08:30Z and
every hour back to 00:30Z). The ≈$41-of-$50 figure is still 2026-09-28 03:30Z's **extrapolation,
not a reading**; Vast `/users/current/` exposes no balance field.

**Item 43 (open, dated):** idle-storage-reaches-$50 projection ≈2026-10-02. The re-notify is
owed on **2026-10-01** and **not before** — today is 09-29, so **not re-notified this hour**,
deliberately. Do not notify early; do not notify more than once. `47913006` remains
**stop-only, never destroy** (sole copy of the 213 GB family-5 daily tensor).

**Standing items.** **41 still open, BLOCKING** — the session's own project-doc listing still
shows **660 docs with `claude/fleet-check-2026-09-25-1530Z.md` newest**, i.e. no project write has
landed in four days and the store is unpruned; `project_info` was NOT called this hour (the
listing already answers it read-only). This file remains the record; nothing deleted, which stays
Chris's call. 42 open and it fired this hour (above). 40, 37, 31, 16 unchanged. 15 — **121st**
fresh-container bootstrap; the container had no `earth`, no `.gh_pat`, no `.vast_key`. Re-cloned at
`19b5046` and rewrote both credential files from `claude/github-access.md` Rule 2b +
`claude/vast-access.md` with the **Write tool, never argv**, `chmod 600`. The first
`git clone --depth 1 --filter=blob:none` succeeded first try this time. Routine, not an event.
Mapping unchanged (4): `47913006`←`gpu-box-46996216`, `49102182`←`gpu-box-31299601`,
`50928407`←`gpu-box-46694776`, `51415980`←`gpu-box-31947967`.

**The telemetry note holds and was used:** `fleet_box_detail.mjs` prints no `gpu_util`,
`cpu_util`, `gpu_temp`, `vmem` or `disk_util` — the per-condition telemetry above is a direct
Node `fetch` of `GET /api/v1/instances/` (**v1, not v0**; v0 returns an empty `instances` array
with this key, which looks exactly like "no boxes exist").

**CPU-BOUND rule unchanged:** decide against a control's first `stage2_step` `wall_s`
(`#478`, K=144/1024x16/batch256 → `wall_s 240`), **never a threshold**; exclude the k-fold ridge
solves and a run's first 1–2 h of anomaly transform + embed; **name the script**; write the
deadline and threshold down before the evidence closes; price both errors. Never destroy from
this session; never stop a box with a running job.

**For 11:30Z:** nothing in flight on the fleet, no run to deadline, no box to watch. Expect this
commit's own `Test & Deploy` as a new unfinished hosted run, failing in 45–58 min — known pattern,
not a finding. **Check item 42 first**: if the entry directly below is not 10:30Z, the schedule
skipped again; two consecutive skips across an idle fleet is still not a notify, but note the run
of them. Append here rather than creating a new file.

**No notification sent** — healthy, unchanged, the hour's only closed item (`#1252`) landed inside
its known band, and the one skipped check happened over an idle fleet. See item 43 for the one
dated thing that warrants a ping on 10-01.

---

## 2026-09-29 08:30Z — HEALTHY (exit 0, twice), fleet unchanged from 07:30Z. 0 mutations, 0 fleet commits.

`fleet_health.mjs`: `1 run(s) not finished · 0 runner(s) online+idle` / HEALTHY / EXIT=0 at
08:31:1xZ and again at 08:35:3xZ.

**Item 42 checked first, per 07:30Z's instruction: did NOT recur** — the entry directly below
is 07:30Z, so the schedule held a second consecutive hour. Skip history unchanged: 06:30Z/07:30Z,
13:30Z/14:30Z, 22:30Z (all 09-28), 05:30Z (09-29).

**All five conditions clean; every Vast field byte-identical to 07:30Z, 06:30Z and 04:30Z, across
two frames 124 s apart (08:31:51Z and 08:33:55Z):**

- **CPU-BOUND** n/a — no box `running`, no job anywhere on the fleet, 0 runners online.
  `gpu_util` and `cpu_util` 0 on all four. No threshold used, no control read, no script to
  name, no deadline to write down.
- **IDLE BURN** clean — 4 instances, all `cur_state=stopped` AND `intended_status=stopped`.
  **$0.00/h GPU burn.** No `gpu_box.mjs stop` issued.
- **QUEUE STALL** cannot fire — `in_progress` is **empty (0 runs)**; `queued` is exactly
  `#650 Test & Deploy`, queued 2026-08-19T04:04:09Z (item 16 baseline). 0 runners online,
  nothing to pair it with. The single "run not finished" is `#650` and nothing else.
- **DISK** clean — `50928407` 132/200 = **66.0%** (highest), `51415980` 184/420 = 43.8%,
  `49102182` 89/300 = 29.7%, `47913006` 0.8/700 = 0.1%. All far under 90%.
- **TELEMETRY** clean — `gpu_temp` 60.0 / 38.0 / 49.0 / 53.0 °C in both frames, `vmem`
  0.473 / 0.471 / 0.340 / 0.369. No dead frame. `50928407` still `stopped`/`loading`, the
  stale pair first noted 2026-09-25 12:30Z — thirteenth day, not billing GPU, not actioned.

**07:30Z's one prediction landed exactly as predicted, so it is not a finding.** `#1251 Test &
Deploy` — spawned by the 07:30Z check's own commit `fac5911` — ran 07:34:24Z → 08:30:40Z
(**56.3 min**) and ended `failure`. **Inside the 45–58 min band, now unbroken across #1247
(45.5 m), #1249 (56.9 m), #1250 (51.7 m) and #1251 (56.3 m), all `failure`.** Test & Deploy has
been red for days; **still noted, still not actioned, still not new.** It finished ~6 min before
this check, so again there is **no unfinished hosted run at all** — which is why the count read 1
and not 2. Other hosted lanes unchanged and green since 07:30Z (`#61 slatrack fetch`
06:11:39Z→06:18:36Z success remains the last; `#100 GLORYS pull (global)` 05:49Z, `#168 GLORYS
pull` 04:38Z, `#223 tpu-status-mirror` 02:37Z all success; nothing new has fired). All
`ubuntu-latest`, **no fleet box, no Vast cost. Runner registrations 19, unchanged** (21 on
09-25 → 20 → 19), all `offline busy=false`. Nothing is meant to be in flight, and nothing is.

**Budget unchanged:** $0.00/h GPU; **$0.0815/h storage = $1.96/day** across the four stopped
boxes (`storage_total_cost` 0.012963 + 0.011111 + 0.018519 + 0.038889 — identical to 07:30Z and
every hour back to 00:30Z). The ≈$41-of-$50 figure is still 2026-09-28 03:30Z's **extrapolation,
not a reading**; Vast `/users/current/` exposes no balance field.

**Item 43 (open, dated):** idle-storage-reaches-$50 projection ≈2026-10-02. The re-notify is
owed on **2026-10-01** and **not before** — today is 09-29, so **not re-notified this hour**,
deliberately. Do not notify early; do not notify more than once. `47913006` remains
**stop-only, never destroy** (sole copy of the 213 GB family-5 daily tensor).

**Standing items.** **41 still open, BLOCKING** — the session's own project-doc listing still
shows **660 docs with `claude/fleet-check-2026-09-25-1530Z.md` newest**, i.e. no project write
has landed in four days and the store is unpruned; `project_info` was NOT called this hour
(the listing already answers it read-only, and the call is not free). This file remains the
record; nothing deleted, which stays Chris's call. 42 open but quiet this hour (above). 40, 37,
31, 16 unchanged. 15 — **120th** fresh-container bootstrap; the container had no `earth`, no
`.gh_pat`, no `.vast_key`. Re-cloned at `fac5911` and rewrote both credential files from
`claude/github-access.md` Rule 2b + `claude/vast-access.md` with the **Write tool, never argv**,
`chmod 600`. Routine, not an event. Mapping unchanged (4): `47913006`←`gpu-box-46996216`,
`49102182`←`gpu-box-31299601`, `50928407`←`gpu-box-46694776`, `51415980`←`gpu-box-31947967`.

**The note from 07:30Z holds and was used:** `fleet_box_detail.mjs` prints no `gpu_util`,
`cpu_util`, `gpu_temp`, `vmem` or `disk_util` — the per-condition telemetry above is a direct
Node `fetch` of `GET /api/v1/instances/` (**v1, not v0**; v0 returns an empty `instances` array
with this key, which looks exactly like "no boxes exist").

**CPU-BOUND rule unchanged:** decide against a control's first `stage2_step` `wall_s`
(`#478`, K=144/1024x16/batch256 → `wall_s 240`), **never a threshold**; exclude the k-fold ridge
solves and a run's first 1–2 h of anomaly transform + embed; **name the script**; write the
deadline and threshold down before the evidence closes; price both errors. Never destroy from
this session; never stop a box with a running job.

**For 09:30Z:** nothing in flight on the fleet, no run to deadline, no box to watch. Expect this
commit's own `Test & Deploy` (**`#1252`**) as a new unfinished hosted run, failing in 45–58 min
— known pattern, not a finding. **Check item 42 first**: if the entry directly below is not
08:30Z, the schedule skipped; a skip across an idle fleet is still not a notify, a skip with a
box `running` is. Append here rather than creating a new file.

**No notification sent** — healthy, unchanged, and the hour's only closed item (`#1251`) landed
inside its known band. See item 43 for the one dated thing that warrants a ping on 10-01.

---

## 2026-09-29 07:30Z — HEALTHY (exit 0), fleet unchanged from 06:30Z. 0 mutations, 0 fleet commits.

`fleet_health.mjs`: `1 run(s) not finished · 0 runner(s) online+idle` / HEALTHY / EXIT=0 at
07:30:5xZ.

**Item 42 did NOT recur this hour** — the entry directly below is 06:30Z, so the schedule held.
Checked first, per 06:30Z's instruction. Skip history unchanged: 06:30Z/07:30Z, 13:30Z/14:30Z,
22:30Z (all 09-28), 05:30Z (09-29).

**All five conditions clean; every Vast field byte-identical to 06:30Z and 04:30Z, across two
frames 145 s apart (07:31:03Z and 07:33:28Z):**

- **CPU-BOUND** n/a — no box `running`, no job anywhere on the fleet, 0 runners online.
  `gpu_util` and `cpu_util` 0 on all four. No threshold used, no control read, no script to
  name, no deadline to write down.
- **IDLE BURN** clean — 4 instances, all `cur_state=stopped` AND `intended_status=stopped`.
  **$0.00/h GPU burn.** No `gpu_box.mjs stop` issued.
- **QUEUE STALL** cannot fire — `in_progress` is **empty (0 runs)**; `queued` is exactly
  `#650 Test & Deploy`, queued 2026-08-19T04:04:09Z (item 16 baseline). 0 runners online,
  nothing to pair it with. The single "run not finished" is `#650` and nothing else.
- **DISK** clean — `50928407` 132/200 = **66.0%** (highest), `51415980` 184/420 = 43.8%,
  `49102182` 89/300 = 29.7%, `47913006` 0.8/700 = 0.1%. All far under 90%.
- **TELEMETRY** clean — `gpu_temp` 60.0 / 38.0 / 49.0 / 53.0 °C in both frames, `vmem`
  0.473 / 0.471 / 0.340 / 0.369. No dead frame. `50928407` still `stopped`/`loading`, the
  stale pair first noted 2026-09-25 12:30Z — twelfth day, not billing GPU, not actioned.

**06:30Z's one prediction landed exactly as predicted, so it is not a finding.** `#1250 Test &
Deploy` — spawned by the 06:30Z check's own commit `1e8891f` — ran 06:35:46Z → 07:27:30Z
(**51.7 min**) and ended `failure`. **Inside the 45–58 min band, which is now unbroken across
#1247 (45.5 m), #1249 (56.9 m) and #1250 (51.7 m), all `failure`.** Test & Deploy has been red
for days; **still noted, still not actioned, still not new.** It finished before this check, so
unlike the last several hours there is **no unfinished hosted run at all** — that is why the
count read 1 and not 2. Other hosted lanes unchanged since 06:30Z (`#61 slatrack fetch`
06:11:39Z→06:18:36Z success was the last; nothing new has fired). All `ubuntu-latest`, **no
fleet box, no Vast cost. Runner registrations 19, unchanged** (21 on 09-25 → 20 → 19), all
`offline busy=false`. Newest `ml-train` remains `#554`, 2026-09-07. Nothing is meant to be in
flight, and nothing is.

**Budget unchanged:** $0.00/h GPU; **$0.0815/h storage = $1.96/day** across the four stopped
boxes (`storage_total_cost` 0.012963 + 0.011111 + 0.018519 + 0.038889 — identical to 06:30Z and
every hour back to 00:30Z). The ≈$41-of-$50 figure is still 2026-09-28 03:30Z's **extrapolation,
not a reading**; Vast `/users/current/` exposes no balance field.

**Item 43 (open, dated):** idle-storage-reaches-$50 projection ≈2026-10-02. The re-notify is
owed on **2026-10-01** and **not before** — today is 09-29, so **not re-notified this hour**,
deliberately. Do not notify early; do not notify more than once. `47913006` remains
**stop-only, never destroy** (sole copy of the 213 GB family-5 daily tensor).

**Standing items.** **41 still open, BLOCKING** — re-checked read-only via `project_info`:
`knowledge_size` **2,000,415 / 2,000,000**, byte-identical to 06:30Z, 04:30Z and 03:30Z, still
*over* the cap; newest project doc still `claude/fleet-check-2026-09-25-1530Z.md`, so no project
write has landed in four days. This file remains the record; nothing deleted, which stays
Chris's call. 42 open but quiet this hour (above). 40, 37, 31, 16 unchanged. 15 — **119th**
fresh-container bootstrap; the container had no `earth`, no `.gh_pat`, no `.vast_key`. Re-cloned
at `1e8891f` and rewrote both credential files from `claude/github-access.md` Rule 2b +
`claude/vast-access.md` with the **Write tool, never argv**, `chmod 600`. Routine, not an event.
Mapping unchanged (4): `47913006`←`gpu-box-46996216`, `49102182`←`gpu-box-31299601`,
`50928407`←`gpu-box-46694776`, `51415980`←`gpu-box-31947967`.

**Note for a future check:** `fleet_box_detail.mjs` does **not** print `gpu_util`, `cpu_util`,
`gpu_temp`, `vmem` or `disk_util` — only state, cost and hardware. The per-condition telemetry
above comes from a direct Node `fetch` of `GET /api/v1/instances/` (**v1, not v0** — v0 returns
an empty `instances` array with this key, which looks exactly like "no boxes exist"). Worth
knowing before concluding the fleet is empty.

**CPU-BOUND rule unchanged:** decide against a control's first `stage2_step` `wall_s`
(`#478`, K=144/1024x16/batch256 → `wall_s 240`), **never a threshold**; exclude the k-fold ridge
solves and a run's first 1–2 h of anomaly transform + embed; **name the script**; write the
deadline and threshold down before the evidence closes; price both errors. Never destroy from
this session; never stop a box with a running job.

**For 08:30Z:** nothing in flight on the fleet, no run to deadline, no box to watch. Expect this
commit's own `Test & Deploy` (**`#1251`**) as a new unfinished hosted run, failing in 45–58 min
— known pattern, not a finding. **Check item 42 first**: if the entry directly below is not
07:30Z, the schedule skipped; a skip across an idle fleet is still not a notify, a skip with a
box `running` is. Append here rather than creating a new file.

**No notification sent** — healthy, unchanged, and the hour's only closed item (`#1250`) landed
inside its known band. See item 43 for the one dated thing that warrants a ping on 10-01.

---

## 2026-09-29 06:30Z — HEALTHY (exit 0, twice), fleet unchanged from 04:30Z. 0 mutations, 0 fleet commits. **The 05:30Z check never ran.**

`fleet_health.mjs`: `1 run(s) not finished · 0 runner(s) online+idle` / HEALTHY / EXIT=0, at
06:31:20Z and again at 06:35:50Z. **This entry covers two hours**, because the entry directly
below is 04:30Z.

**Item 42 DID recur, and the evidence is positive rather than an absence.** No 05:30Z entry
in this file, and the last commit on `main` is the 04:30Z check's own `f67a168`. The
independent confirmation is `#1249 Test & Deploy`: it was created 04:35:47Z **on sha
`f67a168`**, and it is the newest `Test & Deploy` in the repo — had a 05:30Z check run, its
commit would have spawned a run of its own and concurrency-cancelled `#1249`. So the watch
was dark from 04:36Z to 06:31Z, **~1 h 55 m**. **Not notified**, on the 2026-09-28 23:30Z
rule and for the same specific reason: the fleet was *fully idle* across the whole hole — 4
boxes `stopped`, no job anywhere, **$0.00/h GPU** — so the unwatched hour cost nothing and
could not have hidden a CPU-BOUND, IDLE BURN or QUEUE STALL. Item 42 stays **open**; the
hole that *is* worth waking Chris for is one that lands while a box reads `running`. The
skip pattern so far: 06:30Z/07:30Z, 13:30Z/14:30Z, 22:30Z (all 09-28), now 05:30Z (09-29).

**All five conditions clean; every Vast field byte-identical to 04:30Z, across two frames
127 s apart (06:31:57Z and 06:34:04Z):**

- **CPU-BOUND** n/a — no box `running`, no job anywhere on the fleet, 0 runners online.
  `gpu_util` and `cpu_util` 0 on all four. No threshold used, no control read, no script to
  name, no deadline to write down.
- **IDLE BURN** clean — 4 instances, all `cur_state=stopped` AND `intended_status=stopped`.
  **$0.00/h GPU burn.** No `gpu_box.mjs stop` issued.
- **QUEUE STALL** cannot fire — `in_progress` is **empty (0 runs)**; `queued` is exactly
  `#650 Test & Deploy`, queued 2026-08-19T04:04:09Z (item 16 baseline). 0 runners online,
  nothing to pair it with. The 2→1 drop in "runs not finished" is `#1248`/`#1249` closing,
  below — not a fleet event.
- **DISK** clean — `50928407` 132/200 = **66.0%** (highest), `51415980` 184/420 = 43.8%,
  `49102182` 89/300 = 29.7%, `47913006` 0.75/700 = 0.1%. All far under 90%.
- **TELEMETRY** clean — `gpu_temp` 60.0 / 38.0 / 49.0 / 53.0 °C in both frames. No dead
  frame. `50928407` still `stopped`/`loading`, the stale pair first noted 2026-09-25 12:30Z
  — twelfth day, not billing GPU, not actioned.

**The one thing 04:30Z deferred is CLOSED, and it was not a hang.** `#1248` went
`completed/cancelled` at 04:36:10Z — **concurrency-cancelled by `#1249`** on the 04:30Z
check's own commit, exactly the way `#1229` was by `#1230` and `#1227` by `#1228`. It had
reached 60.4 min, so the reading "marginally past the 45–58 min band" was right and the
question of whether it would hang answered itself. **Its successor `#1249` ran 04:35:47Z →
05:32:43Z (56.9 min) and ended `failure` — inside the band, at the top.** So the band does
**not** need re-stating: 45–58 min, all `failure`, unbroken. Test & Deploy has been red for
days; **still noted, still not actioned, still not new.** All three other hosted lanes green:
`#168 GLORYS pull` 04:38:47Z→04:49:52Z, `#100 GLORYS pull (global)` 05:49:02Z→05:55:23Z,
`#61 slatrack fetch 1993-2024` 06:11:39Z→06:18:36Z (**third consecutive success** — the
3-of-14 intermittency noted 09-29 03:30Z continues to clear). All `ubuntu-latest`, **no
fleet box, no Vast cost. Runner registrations 19, unchanged** (21 on 09-25 → 20 → 19), all
`offline busy=false`. Newest `ml-train` remains `#554`, 2026-09-07. Nothing is meant to be
in flight, and nothing is.

**Budget unchanged:** $0.00/h GPU; **$0.0815/h storage = $1.96/day** across the four stopped
boxes (`storage_total_cost` 0.012963 + 0.011111 + 0.018519 + 0.038889 — identical to 04:30Z
and every hour back to 00:30Z). The ≈$41-of-$50 figure is still 2026-09-28 03:30Z's
**extrapolation, not a reading**; Vast `/users/current/` exposes no balance field.

**Item 43 (open, dated):** idle-storage-reaches-$50 projection ≈2026-10-02. The re-notify is
owed on **2026-10-01** and **not before** — today is 09-29, so **not re-notified this hour**,
deliberately. Do not notify early; do not notify more than once. `47913006` remains
**stop-only, never destroy** (sole copy of the 213 GB family-5 daily tensor).

**Standing items.** **41 still open, BLOCKING** — re-checked read-only via `project_info`:
`knowledge_size` **2,000,415 / 2,000,000**, byte-identical to 04:30Z and 03:30Z, still *over*
the cap; newest project doc still `claude/fleet-check-2026-09-25-1530Z.md`, so no project
write has landed in four days. This file remains the record; nothing deleted, which stays
Chris's call. 42 open and recurring (above). 40, 37, 31, 16 unchanged. 15 — **118th**
fresh-container bootstrap; the container had no `earth`, no `.gh_pat`, no `.vast_key`.
Re-cloned at `f67a168` and rewrote both credential files from `claude/github-access.md`
Rule 2b + `claude/vast-access.md` with the **Write tool, never argv**, `chmod 600`. Routine,
not an event. Mapping unchanged (4): `47913006`←`gpu-box-46996216`,
`49102182`←`gpu-box-31299601`, `50928407`←`gpu-box-46694776`, `51415980`←`gpu-box-31947967`.

**CPU-BOUND rule unchanged:** decide against a control's first `stage2_step` `wall_s`
(`#478`, K=144/1024x16/batch256 → `wall_s 240`), **never a threshold**; exclude the k-fold
ridge solves and a run's first 1–2 h of anomaly transform + embed; **name the script**; write
the deadline and threshold down before the evidence closes; price both errors. Never destroy
from this session; never stop a box with a running job.

**For 07:30Z:** nothing in flight on the fleet, no run to deadline, no box to watch. Expect
this commit's own `Test & Deploy` (**`#1250`**) as the new unfinished hosted run, failing in
45–58 min — known pattern, not a finding. **Check item 42 first**: if the entry directly
below this one is not 06:30Z, the schedule skipped again; two skips inside a single UTC day
while the fleet is idle is still not a notify, but a skip with a box `running` is. Watch
whether runner registrations keep shedding (19 now) — benign at any count while `#650` is the
only queued run. Append here rather than creating a new file.

**No notification sent** — healthy, and the two deltas are a known-pattern schedule skip
across a fully idle fleet and a hosted CI pair closing inside its usual band. See item 43 for
the one dated thing that warrants a ping on 10-01.

---

## 2026-09-29 04:30Z — HEALTHY (exit 0, twice), fleet unchanged from 03:30Z. 0 mutations, 0 fleet commits.

`fleet_health.mjs`: `2 run(s) not finished · 0 runner(s) online+idle` / HEALTHY / EXIT=0, at
04:31:24Z and again at 04:34:21Z. **Item 42 did NOT recur** — 03:30Z is the entry directly
below, so the schedule has now held five hours running (00:30Z–04:30Z).

**The "2 not finished" is 1→2 and is NOT a change of state:** it is `#650` plus `#1248`,
the hosted `Test & Deploy` on the 03:30Z check's own commit `602282f`, which the 03:30Z
entry predicted by number. `in_progress` = exactly `#1248` (created 03:35:46Z,
`ubuntu-latest`, **no fleet box, no Vast cost**); `queued` = exactly `#650`.

**All five conditions clean; every Vast field byte-identical to 03:30Z, across two frames
~135 s apart (04:31:59Z and 04:34:16Z):**

- **CPU-BOUND** n/a — no box `running`, no job anywhere on the fleet, 0 runners online.
  `gpu_util` and `cpu_util` 0 on all four. No threshold used, no control read, no script
  to name.
- **IDLE BURN** clean — 4 instances, all `cur_state=stopped` AND `intended_status=stopped`.
  **$0.00/h GPU burn.** No `gpu_box.mjs stop` issued.
- **QUEUE STALL** cannot fire — the queued run IS `#650 Test & Deploy`, queued
  2026-08-19T04:04:09Z (item 16 baseline); 0 runners online, nothing to pair it with.
- **DISK** clean — `50928407` 132/200 = **66.0%** (highest), `51415980` 184/420 = 43.8%,
  `49102182` 89/300 = 29.7%, `47913006` 0.75/700 = 0.1%. All far under 90%.
- **TELEMETRY** clean — `gpu_temp` 60.0 / 38.0 / 49.0 / 53.0 °C in both frames. No dead
  frame. `50928407` still `stopped`/`loading`, the stale pair first noted 2026-09-25
  12:30Z — eleventh day, not billing GPU, not actioned.

**The hour's deltas are zero on the fleet and one off it.** `#1248` was still `in_progress`
at 04:35:07Z, **59.4 min in — marginally PAST the top of the 45–58 min band** (#1247 45.5,
#1246 44.6, #1245 45.5, #1244 58.0). Called **not a finding**, on three grounds: it is
GitHub-hosted `ubuntu-latest`, so it spends **no Vast credit and touches no fleet box**;
the band is a soft observed range that has already drifted once (2026-09-28 12:30Z widened
it); and a hosted run's own 6 h timeout bounds the worst case at zero cost to the $50 cap.
**Left running deliberately — nothing was cancelled.** If 05:30Z finds `#1248` still
`in_progress` (>2 h), that IS a new pattern and worth reporting; a `failure` at 60–70 min
is just the band widening again. **Runner registrations 19, unchanged**
(21 on 09-25 → 20 → 19); all `offline busy=false`. Newest `ml-train` remains `#554`,
2026-09-07. Nothing is meant to be in flight, and nothing is.

**Budget unchanged:** $0.00/h GPU; **$0.0815/h storage = $1.96/day** across the four
stopped boxes (`storage_total_cost` 0.012963 + 0.011111 + 0.018519 + 0.038889 — identical
to 03:30Z, 02:30Z, 01:30Z and 00:30Z). The ≈$41-of-$50 figure is still 2026-09-28 03:30Z's
**extrapolation, not a reading**.

**Item 43 (open, dated):** idle-storage-reaches-$50 projection ~3 days out (≈2026-10-02).
The re-notify is owed on **2026-10-01** and **not before** — today is 09-29, so **not
re-notified this hour**, deliberately. Do not notify early; do not notify more than once.
`47913006` remains **stop-only, never destroy** (sole copy of the 213 GB family-5 daily
tensor).

**Standing items.** **41 still open, BLOCKING** — re-checked read-only via `project_info`:
`knowledge_size` **2,000,415 / 2,000,000**, byte-identical to 03:30Z, still *over* the cap;
newest project doc still `claude/fleet-check-2026-09-25-1530Z.md`, so no write has landed
in four days. This file remains the record; nothing deleted, which stays Chris's call.
42, 40, 37, 31, 16 unchanged. 15 — **117th** fresh-container bootstrap; re-cloned `earth`
(HEAD `602282f`) and rewrote `.gh_pat` + `.vast_key` from `claude/github-access.md` Rule 2b
+ `claude/vast-access.md` with the **Write tool, never argv**, `chmod 600`. Routine, not an
event. Mapping unchanged (4): `47913006`←`gpu-box-46996216`, `49102182`←`gpu-box-31299601`,
`50928407`←`gpu-box-46694776`, `51415980`←`gpu-box-31947967`.

**CPU-BOUND rule unchanged:** decide against a control's first `stage2_step` `wall_s`
(`#478`, K=144/1024x16/batch256 → `wall_s 240`), **never a threshold**; exclude the k-fold
ridge solves and a run's first 1–2 h of anomaly transform + embed; **name the script**;
write the deadline and threshold down before the evidence closes; price both errors. Never
destroy from this session; never stop a box with a running job.

**For 05:30Z:** nothing in flight on the fleet, no run to deadline, no box to watch.
Expect `#1248` closed as `failure` (it was already 59.4 min in at 04:35Z, so 60–75 min
total is the likely figure — record it, the band needs re-stating) and this commit's own
run `#1249` to be the new unfinished hosted run. **If `#1248` is still `in_progress`, that
is the one thing this hour deferred to you — report it.** Watch
whether runner registrations keep shedding (19 now); benign at any count while `#650` is
the only queued run. Append here rather than creating a new file.

**No notification sent** — healthy, unchanged, and the hour's only motion is a hosted CI
run still inside its usual band. See item 43 for the one dated thing that warrants a ping
on 10-01.

---

## 2026-09-29 03:30Z — HEALTHY (exit 0, twice), fleet unchanged from 02:30Z. 0 mutations, 0 fleet commits.

`fleet_health.mjs`: `1 run(s) not finished · 0 runner(s) online+idle` / HEALTHY / EXIT=0, at
03:31:05Z and again at 03:34:16Z. **Item 42 did NOT recur** — 02:30Z is the entry directly
below, so the schedule held again.

**All five conditions clean; every Vast field byte-identical to 02:30Z, across two frames
~110 s apart (03:32:28Z and 03:34:16Z):**

- **CPU-BOUND** n/a — no box `running`, no job anywhere on the fleet, 0 runners online.
  `gpu_util` and `cpu_util` 0 on all four. No threshold used, no control read, no script
  to name.
- **IDLE BURN** clean — 4 instances, all `cur_state=stopped` AND `intended_status=stopped`.
  **$0.00/h GPU burn.** No `gpu_box.mjs stop` issued.
- **QUEUE STALL** cannot fire — the one unfinished run IS `#650 Test & Deploy`, queued
  2026-08-19T04:04:09Z (item 16 baseline); 0 runners online, nothing to pair it with.
  `in_progress` is **empty** (0 runs); `queued` is exactly `#650`.
- **DISK** clean — `50928407` 132/200 = **66.0%** (highest), `51415980` 184/420 = 43.8%,
  `49102182` 89/300 = 29.7%, `47913006` 0.75/700 = 0.1%. All far under 90%.
- **TELEMETRY** clean — `gpu_temp` 60.0 / 38.0 / 49.0 / 53.0 °C in both frames. No dead
  frame. `50928407` still `stopped`/`loading`, the stale pair first noted 2026-09-25
  12:30Z — tenth day, not billing GPU, not actioned.

**The hour's deltas are two, neither a fleet event.** (1) `#1247 Test & Deploy` on sha
`dd9fd38` — the 02:30Z check's own commit — ran 02:39:47Z → **failure 03:25:19Z, ~45.5 min**,
at the bottom of the 45–58 min band; GitHub-hosted `ubuntu-latest`, **no fleet box, no Vast
cost**, the standing `app.spec.js` red, same as #1246/1245/1244/1243/1242/1241/1239/1238.
Predicted verbatim by the 02:30Z entry, so not a finding. (2) **Runner registrations 19,
was 20** — one more stale offline registration pruned (21 on 09-25 → 20 → 19). All 19 are
`offline busy=false`; an offline registration bills nothing and the only queued run is
`#650`, so this is GitHub housekeeping, not a loss. Newest `ml-train` remains `#554`,
2026-09-07. Nothing is meant to be in flight, and nothing is.

**Budget unchanged:** $0.00/h GPU; **$0.0815/h storage = $1.96/day** across the four stopped
boxes (`storage_total_cost` 0.012963 + 0.011111 + 0.018519 + 0.038889 — identical to 02:30Z,
01:30Z and 00:30Z). The ≈$41-of-$50 figure is still 2026-09-28 03:30Z's **extrapolation, not
a reading**.

**Item 43 (open, dated):** the idle-storage-reaches-$50 projection is now **~3 days out
(≈2026-10-02)**. Per 02:30Z's recommendation the re-notify is owed on **2026-10-01** and
**not before** — today is 09-29, so **not re-notified this hour**, deliberately. Do not
notify early; do not notify more than once. `47913006` remains **stop-only, never destroy**
(sole copy of the 213 GB family-5 daily tensor).

**Standing items.** **41 still open, BLOCKING** — re-checked read-only this hour via
`project_info`: `knowledge_size` **2,000,415 / 2,000,000**, i.e. still *over* the cap, and
the newest project doc is still `claude/fleet-check-2026-09-25-1530Z.md`, so no write has
landed in four days. 02:30Z's finding stands (a ~137-token write was refused; the cap is
total, not marginal). This file remains the record; nothing deleted, which stays Chris's
call. 42, 40, 37, 31, 16 unchanged. 15 — **116th** fresh-container bootstrap; re-cloned
`earth` (`--depth 1 --filter=blob:none`, seconds, first attempt; HEAD `dd9fd38`) and rewrote
`.gh_pat` + `.vast_key` from `claude/github-access.md` Rule 2b + `claude/vast-access.md` with
the **Write tool, never argv**, `chmod 600`. Routine, not an event. Mapping unchanged (4):
`47913006`←`gpu-box-46996216`, `49102182`←`gpu-box-31299601`,
`50928407`←`gpu-box-46694776`, `51415980`←`gpu-box-31947967`.

**CPU-BOUND rule unchanged:** decide against a control's first `stage2_step` `wall_s`
(`#478`, K=144/1024x16/batch256 → `wall_s 240`), **never a threshold**; exclude the k-fold
ridge solves and a run's first 1–2 h of anomaly transform + embed; **name the script**;
write the deadline and threshold down before the evidence closes; price both errors. Never
destroy from this session; never stop a box with a running job.

**For 04:30Z:** nothing in flight on the fleet, no run to deadline, no box to watch. Expect
this commit's own `Test & Deploy` (`#1248`) as an unfinished hosted run, failing in ~45–58
min — known pattern, not a finding. Watch whether runner registrations keep shedding (19
now); still benign at any count while `#650` is the only queued run. Append here rather than
creating a new file.

**No notification sent** — healthy, unchanged, and the hour's two deltas are a hosted CI run
failing on the standing red exactly as the last eight did, plus one free offline runner
registration expiring. See item 43 for the one dated thing that warrants a ping on 10-01.

---

## 2026-09-29 02:30Z — HEALTHY (exit 0), fleet unchanged from 01:30Z. 0 mutations, 0 fleet commits.

`fleet_health.mjs`: `1 run(s) not finished · 0 runner(s) online+idle` / HEALTHY / EXIT=0.
**Item 42 did NOT recur** — 01:30Z is the entry directly below, so the schedule held again.

**All five conditions clean; every Vast field byte-identical to 01:30Z, across two frames
60 s apart (02:37:04Z and 02:38:28Z):**

- **CPU-BOUND** n/a — no box `running`, no job anywhere on the fleet, 0 runners online.
  `gpu_util` and `cpu_util` 0 on all four. No threshold used, no control read, no script
  to name.
- **IDLE BURN** clean — 4 instances, all `cur_state=stopped` AND `intended_status=stopped`.
  **$0.00/h GPU burn.** No `gpu_box.mjs stop` issued.
- **QUEUE STALL** cannot fire — the one unfinished run IS `#650 Test & Deploy`, queued
  2026-08-19T04:04:09Z (item 16 baseline); 0 runners online, nothing to pair it with. The
  last 15 GitHub runs are all `completed`; `in_progress` is **empty**.
- **DISK** clean — `50928407` 132/200 = **66.0%** (highest), `51415980` 184/420 = 43.8%,
  `49102182` 89/300 = 29.7%, `47913006` 0.75/700 = 0.1%. All far under 90%.
- **TELEMETRY** clean — `gpu_temp` 60.0 / 38.0 / 49.0 / 53.0 °C in both frames. No dead
  frame. `50928407` still `stopped`/`loading`, the stale pair first noted 2026-09-25
  12:30Z — ninth day, not billing GPU, not actioned.

**The hour's one delta is a hosted CI run, exactly as 01:30Z predicted.** `#1246 Test &
Deploy` on sha `35ca441` — the 01:30Z check's own commit — ran 01:34:39Z → **failure
02:19:14Z, ~44.6 min**, just under the 46–58 min band of its predecessors. GitHub-hosted,
**no fleet box, no Vast cost**; the standing `app.spec.js` red at step 6, same as
#1245/1244/1243/1242/1241/1239/1238. It cancelled nothing. Newest `ml-train` remains
`#554`, 2026-09-07. Nothing is meant to be in flight, and nothing is.

**Runner registrations 20, unchanged.** All offline, 0 busy.

**Budget unchanged:** $0.00/h GPU; **$0.0815/h storage = $1.96/day** across the four
stopped boxes (`storage_total_cost` 0.012963 + 0.011111 + 0.018519 + 0.038889 — identical
to 01:30Z and 00:30Z). The ≈$41-of-$50 figure is still 2026-09-28 03:30Z's
**extrapolation, not a reading**.

**Item 43 NEW (open, dated, needs a decision by 10-01):** the idle-storage-reaches-$50
projection is now **~3 days out (≈2026-10-02)**. It was reported once, 09-28 08:30Z, and
has been carried silently for 18 checks on the correct grounds that an unchanged,
already-reported projection is not a reason to wake Chris. **That reasoning expires as the
date closes.** Auto-top-up is ENABLED (`claude/vast-access.md`), so the $50 cap is a
user-approved budget line, not a hard stop — crossing it is a real breach, and the only
levers are Chris's (destroy or shrink a box, or raise the cap). **Recommendation for
whichever check runs on 2026-10-01: re-notify once, even though nothing will have
changed** — that is the last hour at which a heads-up is still actionable a day ahead.
Do not re-notify before then, and do not notify more than once. Note `47913006` is
**stop-only, never destroy** (sole copy of the 213 GB family-5 daily tensor), so it is not
a candidate for the shrink even though its 700 GB is the largest storage line ($0.0389/h,
48% of the idle burn).

**Standing items.** **41 still open, BLOCKING** — re-tested this hour with a deliberately
tiny write: a **~137-token** `project_write` was **refused** ("would exceed the project's
maximum size (~2000000 tokens)"). That is the smallest write attempted so far, so the cap
is total, not marginal — **no compaction of the hourly entry can get under it.** This file
remains the record; nothing deleted, which stays Chris's call. 42, 40, 37, 31, 16
unchanged. 15 — **115th** fresh-container bootstrap; re-cloned `earth`
(`--depth 1 --filter=blob:none`, seconds, first attempt) and rewrote `.gh_pat` +
`.vast_key` from `claude/github-access.md` Rule 2b + `claude/vast-access.md` with the
**Write tool, never argv**, `chmod 600`. Routine, not an event. Mapping unchanged (4):
`47913006`←`gpu-box-46996216`, `49102182`←`gpu-box-31299601`,
`50928407`←`gpu-box-46694776`, `51415980`←`gpu-box-31947967`.

**CPU-BOUND rule unchanged:** decide against a control's first `stage2_step` `wall_s`
(`#478`, K=144/1024x16/batch256 → `wall_s 240`), **never a threshold**; exclude the k-fold
ridge solves and a run's first 1–2 h of anomaly transform + embed; **name the script**;
write the deadline and threshold down before the evidence closes; price both errors. Never
destroy from this session; never stop a box with a running job.

**For 03:30Z:** nothing in flight on the fleet, no run to deadline, no box to watch. Expect
this commit's own `Test & Deploy` as an unfinished hosted run, failing in ~45–58 min —
known pattern, not a finding. Append here rather than creating a new file.

**No notification sent** — healthy, unchanged, and the hour's only delta is a hosted CI run
failing on the standing red exactly as the last seven did. See item 43 for the one dated
thing that will warrant a ping on 10-01.

---

## 2026-09-29 01:30Z — HEALTHY (exit 0), fleet unchanged from 00:30Z. 0 mutations, 0 fleet commits.

`fleet_health.mjs`: `1 run(s) not finished · 0 runner(s) online+idle` / HEALTHY / EXIT=0.
**Item 42 did NOT recur** — 00:30Z is the entry directly below, so the schedule held again.

**All five conditions clean; every Vast field byte-identical to 00:30Z, across two frames
60 s apart (01:31Z and 01:32Z):**

- **CPU-BOUND** n/a — no box `running`, no job anywhere on the fleet, 0 runners online.
  `gpu_util` and `cpu_util` 0 on all four. No threshold used, no control read, no script
  to name.
- **IDLE BURN** clean — 4 instances, all `cur_state=stopped` AND `intended_status=stopped`.
  **$0.00/h GPU burn.** No `gpu_box.mjs stop` issued.
- **QUEUE STALL** cannot fire — the one unfinished run IS `#650 Test & Deploy`, queued
  2026-08-19T04:04:09Z (item 16 baseline); 0 runners online, nothing to pair it with. The
  last 15 GitHub runs are all `completed`.
- **DISK** clean — `50928407` 132/200 = **66.0%** (highest), `51415980` 184/420 = 43.8%,
  `49102182` 89/300 = 29.7%, `47913006` 0.8/700 = 0.1%. All far under 90%.
- **TELEMETRY** clean — `gpu_temp` 60.0 / 38.0 / 49.0 / 53.0 °C in both frames. No dead
  frame. `50928407` still `stopped`/`loading`, the stale pair first noted 2026-09-25
  12:30Z — eighth day, not billing GPU, not actioned.

**The hour's one delta is a hosted CI run, not a fleet event.** `#1245 Test & Deploy`
(`36503701355`) on sha `08dfa6b` — i.e. triggered by the 00:30Z check's own commit —
started 00:33:43Z, ended **failure 01:19:14Z at ~46 min**, inside the expected 46–58 min
band. Both jobs ran on **GitHub-hosted `ubuntu-latest`** (runner `GitHub Actions
1000006808/9`), **no fleet box, no Vast cost**; job `test` failed at step 6 `Run test
suite (data integrity + browser)` — the standing `app.spec.js` red — while job `deploy`
succeeded. Same conclusion and same step as #1244/1243/1242/1241/1239/1238. It cancelled
nothing. Newest `ml-train` remains `#554`, 2026-09-07. Nothing is meant to be in flight,
and nothing is.

**Runner registrations 20, unchanged.** All offline, 0 busy.

**Budget unchanged:** $0.00/h GPU; **$0.0815/h storage = $1.96/day** across the four
stopped boxes (`storage_total_cost` 0.012963 + 0.011111 + 0.018519 + 0.038889 — identical
to 00:30Z). The ≈$41-of-$50 figure is still 2026-09-28 03:30Z's **extrapolation, not a
reading**. Idle disk alone still reaches the $50 cap around **2026-10-02**, now ~3 days
out; reported 09-28 08:30Z, never acted on from this session (§0e), **not re-notified.**

**Standing items.** **41 still open, BLOCKING** — re-tested this hour: a ~869-token
`project_write` of the full record was **refused** ("would exceed the project's maximum
size (~2000000 tokens)"), so this file remains the record and nothing was deleted, which
stays Chris's call. 42, 40, 37, 31, 16 unchanged. 15 — **114th** fresh-container
bootstrap; re-cloned `earth` (`--depth 1 --filter=blob:none`, seconds, first attempt) and
rewrote `.gh_pat` + `.vast_key` from `claude/github-access.md` Rule 2b +
`claude/vast-access.md` with the **Write tool, never argv**, `chmod 600`. Routine, not an
event. Mapping unchanged (4): `47913006`←`gpu-box-46996216`, `49102182`←`gpu-box-31299601`,
`50928407`←`gpu-box-46694776`, `51415980`←`gpu-box-31947967`. `47913006` remains
**stop-only, never destroy** (sole copy of the 213 GB family-5 daily tensor).

**CPU-BOUND rule unchanged:** decide against a control's first `stage2_step` `wall_s`
(`#478`, K=144/1024x16/batch256 → `wall_s 240`), **never a threshold**; exclude the k-fold
ridge solves and a run's first 1–2 h of anomaly transform + embed; **name the script**;
write the deadline and threshold down before the evidence closes; price both errors. Never
destroy from this session; never stop a box with a running job.

**For 02:30Z:** nothing in flight on the fleet, no run to deadline, no box to watch. Expect
this commit's own `Test & Deploy` as an unfinished hosted run, failing in ~50 min — known
pattern, not a finding. Append here rather than creating a new file.

**No notification sent** — healthy, unchanged, and the hour's only delta is a hosted CI run
failing on the standing red exactly as the last six did.

---

## 2026-09-29 00:30Z — HEALTHY (exit 0), fleet unchanged from 23:30Z. 0 mutations, 0 fleet commits.

`fleet_health.mjs`: `1 run(s) not finished · 0 runner(s) online+idle` / HEALTHY / EXIT=0,
twice (00:31Z and 00:34Z). **Item 42 did NOT recur** — 23:30Z is the entry directly below,
so the schedule held this hour.

**All five conditions clean; every Vast field byte-identical to 23:30Z:**

- **CPU-BOUND** n/a — no box `running`, no job anywhere on the fleet, 0 runners online.
  No threshold used, no control read, no script to name.
- **IDLE BURN** clean — 4 instances, all `cur_state=stopped`, `gpu_util` and `cpu_util` 0
  on all four. **$0.00/h GPU burn.** No `gpu_box.mjs stop` issued.
- **QUEUE STALL** cannot fire — the one unfinished run IS `#650 Test & Deploy`, queued
  2026-08-19T04:04:09Z (item 16 baseline); 0 runners online, nothing to pair it with.
- **DISK** clean — `50928407` 132/200 = **66.0%** (highest), `51415980` 184/420 = 43.8%,
  `49102182` 89/300 = 29.7%, `47913006` 0.8/700 = 0.1%. All far under 90%.
- **TELEMETRY** clean — `gpu_temp` 60.0 / 38.0 / 49.0 / 53.0 °C across both frames. No
  dead frame. `50928407` still `stopped`/`loading`, the stale pair first noted
  2026-09-25 12:30Z — seventh day, not billing GPU, not actioned.

**The not-finished count 2 → 1 is last hour's hosted run landing, not a fleet event.**
`#60 slatrack fetch 1993-2024` — the along-track altimetry fetch for family 10.1 — ended
**success 23:33:22Z** after 3.5 min, which is why the count fell back to the `#650`
baseline. `#1244 Test & Deploy`, the 23:30Z check's own commit `996e46a` started
23:32:11Z, ended **failure 00:30:07Z at ~58 min** — inside the expected band and the
expected conclusion (standing red on `app.spec.js`), and it cancelled nothing. Newest
`ml-train` remains `#554`, 2026-09-07. Nothing is meant to be in flight, and nothing is.

**Runner registrations 20, unchanged** after last hour's 21 → 20. All offline, 0 busy.

**Budget unchanged:** $0.00/h GPU; **$0.0815/h storage = $1.96/day** across the four
stopped boxes (`storage_total_cost` 0.012963 + 0.011111 + 0.018519 + 0.038889). The
≈$41-of-$50 figure is still 2026-09-28 03:30Z's **extrapolation, not a reading**. Idle
disk alone still reaches the $50 cap around **2026-10-02** — now ~3 days out; reported
08:30Z, never acted on from this session (§0e), **not re-notified.**

**Standing items.** **41 still open, BLOCKING** — the project store is still over its
2,000,000-token cap, so this file remains the record; nothing deleted, that stays Chris's
call. 42, 40, 37, 31, 16 unchanged. 15 — **113th** fresh-container bootstrap; re-cloned
`earth` and rewrote `.gh_pat` + `.vast_key` from `claude/github-access.md` Rule 2b +
`claude/vast-access.md` with the **Write tool, never argv**, `chmod 600`. Routine, not an
event. Mapping unchanged (4): `47913006`←`gpu-box-46996216`, `49102182`←`gpu-box-31299601`,
`50928407`←`gpu-box-46694776`, `51415980`←`gpu-box-31947967`.

**CPU-BOUND rule unchanged:** decide against a control's first `stage2_step` `wall_s`
(`#478`, K=144/1024x16/batch256 → `wall_s 240`), **never a threshold**; exclude the k-fold
ridge solves and a run's first 1–2 h of anomaly transform + embed; **name the script**;
write the deadline and threshold down before the evidence closes; price both errors. Never
destroy from this session; never stop a box with a running job.

**For 01:30Z:** nothing in flight on the fleet, no run to deadline, no box to watch. Expect
this commit's own `Test & Deploy` as an unfinished hosted run, failing in ~50 min — known
pattern, not a finding. Append here rather than creating a new file.

**No notification sent** — healthy, unchanged, and the one delta (`#60` finishing green) is
a hosted run landing exactly as expected.

---

## 2026-09-28 23:30Z — HEALTHY (exit 0), fleet unchanged from 21:30Z. 0 mutations, 0 fleet commits.

`fleet_health.mjs`: `2 run(s) not finished · 0 runner(s) online+idle` / HEALTHY / EXIT=0.
Two frames — `gpu_box.mjs list` ~23:31Z, then a direct `/api/v1/instances/` read ~23:32Z.

**Item 42 check: the schedule DID skip this hour. The clean streak ended at six.** The
entry immediately below is **21:30Z**, two hours back — **no 22:30Z check ran**, the same
hole shape as 13:30Z/14:30Z and 06:30Z/07:30Z. **Not notified**, and the reason is
specific rather than habit: the fleet was *fully idle* across the missed hour — 4 boxes
stopped, no job anywhere, **$0.00/h GPU** — so an unwatched hour cost nothing and could
not have hidden a CPU-BOUND, IDLE BURN or QUEUE STALL. Item 42 stays **open**; if a hole
ever lands while a box is `running`, that one *is* worth waking Chris for.

**All five conditions clean; every Vast field is byte-identical to 21:30Z:**

- **CPU-BOUND** n/a — no box `running`, no job anywhere on the fleet, 0 runners online.
  No threshold used, no control read, no script to name.
- **IDLE BURN** clean — 4 instances, all `cur_state=stopped`/`intended=stopped`
  (`actual_status` exited/exited/loading/exited), `gpu_util` and `cpu_util` both 0 on all
  four. **$0.00/h GPU burn.** No `gpu_box.mjs stop` issued.
- **QUEUE STALL** cannot fire — `#650 Test & Deploy`, queued 2026-08-19T04:04:09Z, is the
  known baseline (item 16); 0 runners online, so nothing to pair it with.
- **DISK** clean — `50928407` 132/200 = **66.0%** (highest), `51415980` 184/420 = 43.8%,
  `49102182` 89/300 = 29.7%, `47913006` 0.75/700 = 0.1%. All far under 90%.
- **TELEMETRY** clean — `gpu_temp` 60.0 / 38.0 / 49.0 / 53.0 °C, non-zero and identical
  across both frames. No dead frame. `50928407` still `stopped`/`loading`, the stale pair
  first noted 2026-09-25 12:30Z — sixth day, not billing GPU, not actioned.

**The not-finished count 1 → 2 is a new hosted run, not a stall.** `#60 slatrack fetch
1993-2024` started **23:29:52Z** on `8a70bd0` and is `in_progress`; with **0 runners
online** it cannot be a fleet job, so it is `ubuntu-latest` and costs Vast nothing.
`#59` (14:33Z, failure) was the previous newest. `#1243 Test & Deploy` — 21:30Z's own
commit `8a70bd0`, started 21:34:26Z — ended **failure** 22:20:56Z at **~46 min**, inside
the expected band and the expected conclusion (standing red, `app.spec.js`), and unlike
19:30Z/20:30Z it **cancelled nothing**: no concurrency race this hour. Also green and
hosted: `#222 tpu-status-mirror` success 23:24:58Z (was `#221`, 18:20:01Z),
`#99 GLORYS pull (global)` success 23:22:22Z, `#167 GLORYS pull` success 23:02:00Z.
Newest `ml-train` remains `#554`, 2026-09-07. `ml/OVERVIEW.md` still reads
2026-09-16 — nothing new is meant to be in flight, and nothing is.

**Runner registrations 21 → 20.** One registration has gone since 21:30Z. **Not item 37**
(that tracks *instance* departures; all four instances are present and unchanged) and no
impact either way — every runner was already offline, no job is waiting on one, and a
registration costs nothing. Logged as an observation, **not notified**; worth a look only
if the count keeps sliding or a fleet job later fails to find a runner.

**Budget unchanged:** $0.00/h GPU; **$0.0815/h storage = $1.96/day** across the four
stopped boxes, re-summed from `storage_total_cost` (0.012963 + 0.011111 + 0.018519 +
0.038889) — identical to 21:30Z. Per 15:30Z, **do not sum `storage_cost`** — that field is
the host's $/GB/month rate. The ≈$41-of-$50 figure is still 03:30Z's **extrapolation, not a
reading**. Idle disk alone still reaches the $50 cap around **2026-10-02** — now ~4 days
out; reported 08:30Z, per §0e never acted on from this session, **not re-notified.**

**Standing items.** **41 still open, BLOCKING** — the project store is still over its
2,000,000-token cap (660 docs, newest `fleet-check-2026-09-25-1530Z.md`), so this file
remains the record; nothing deleted, that stays Chris's call. **42 open and recurring**
(see above). 40, 37, 31, 16 unchanged. 15 — **112th** fresh-container bootstrap;
re-cloned `earth` and rewrote `.gh_pat` + `.vast_key` from `claude/github-access.md`
Rule 2b + `claude/vast-access.md` with the **Write tool, never argv**, `chmod 600`.
Routine, not an event. Mapping unchanged (4): `47913006`←`gpu-box-46996216`,
`49102182`←`gpu-box-31299601`, `50928407`←`gpu-box-46694776`,
`51415980`←`gpu-box-31947967`.

**CPU-BOUND rule unchanged:** decide against a control's first `stage2_step` `wall_s`
(`#478`, K=144/1024x16/batch256 → `wall_s 240`), **never a threshold**; exclude the k-fold
ridge solves and a run's first 1–2 h of anomaly transform + embed; **name the script**;
write the deadline and threshold down before the evidence closes; price both errors. Never
destroy from this session; never stop a box with a running job.

**For 00:30Z:** nothing in flight on the fleet, no run to deadline, no box to watch.
`#60 slatrack` should have ended; if it is still `in_progress` it is hosted, not a fleet
concern. Append here rather than creating a new file.

**No notification sent** — healthy, unchanged, and the two deltas (a skipped 22:30Z on an
idle fleet, one runner registration) are both zero-cost and already-known shapes.

---

## 2026-09-28 21:30Z — HEALTHY (exit 0), fleet unchanged from 20:30Z. 0 mutations, 0 fleet commits.

`fleet_health.mjs`: `1 run(s) not finished · 0 runner(s) online+idle` / HEALTHY / EXIT=0.
Two frames — `gpu_box.mjs list` ~21:32Z, then a direct `/api/v1/instances/` read ~21:34Z.

**Item 42 check: the schedule did NOT skip this hour.** The entry immediately below is
**20:30Z**, exactly one hour back. **Sixth consecutive clean hour** since the
13:30Z/14:30Z hole. Item 42 stays open (six hours is not a fix) and is **not re-notified**.

**All five conditions clean; every Vast field is byte-identical to 20:30Z:**

- **CPU-BOUND** n/a — no box `running`, no job anywhere on the fleet, 0 of 21 runners
  online. No threshold used, no control read, no script to name.
- **IDLE BURN** clean — 4 instances, all `cur_state=stopped`/`intended=stopped`
  (`actual_status` exited/exited/loading/exited), `gpu_util` and `cpu_util` both 0 on all
  four. **$0.00/h GPU burn.** No `gpu_box.mjs stop` issued.
- **QUEUE STALL** cannot fire — `#650 Test & Deploy`, queued 2026-08-19T04:04:09Z, is the
  known baseline (item 16); 0 runners online, so nothing to pair it with. Still the
  **only** unfinished run: `in_progress` is now **empty** (0 hosted lanes mid-flight), the
  quietest frame in this series.
- **DISK** clean — `50928407` 132/200 = **66.0%** (highest), `51415980` 184/420 = 43.8%,
  `49102182` 89/300 = 29.7%, `47913006` 0.75/700 = 0.1%. All far under 90%.
- **TELEMETRY** clean — `gpu_temp` 60.0 / 38.0 / 49.0 / 53.0 °C, non-zero and identical
  across both frames. No dead frame. `50928407` still `stopped`/`loading`, the stale pair
  first noted 2026-09-25 12:30Z — sixth day, not billing GPU, not actioned.

**No hosted lane opened this hour** — unlike every recent check, this one's predecessor
(`1736328`, 20:32:54Z) is not sitting in a `Test & Deploy`; `in_progress` total is 0. This
commit will start `#1243` and it will very likely end `failure` in ~40–55 min on
`app.spec.js`, which is the standing red, **not a finding**. `#221 tpu-status-mirror`
success 18:20:01Z unchanged; newest `slatrack` remains `#59`; newest `ml-train` remains
`#554`, 2026-09-07.

**Budget unchanged:** $0.00/h GPU; **$0.0815/h storage = $1.96/day** across the four
stopped boxes, re-summed from `storage_total_cost` (0.012963 + 0.011111 + 0.018519 +
0.038889) — identical to 20:30Z. Per 15:30Z, **do not sum `storage_cost`** — that field is
the host's $/GB/month rate. The ≈$41-of-$50 figure is still 03:30Z's **extrapolation, not a
reading**. Idle disk alone still reaches the $50 cap around **2026-10-02**; reported
08:30Z, per §0e never acted on from this session, **not re-notified.**

**Standing items.** **41 still open, BLOCKING** — the project store is still over its
2,000,000-token cap (660 docs listed this hour, newest `fleet-check-2026-09-25-1530Z.md`),
so this file remains the record; nothing deleted, that stays Chris's call. **42 open** (see
above). 40, 37, 31, 16 unchanged. 15 — **111th** fresh-container bootstrap; re-cloned
`earth` and rewrote `.gh_pat` + `.vast_key` from `claude/github-access.md` Rule 2b +
`claude/vast-access.md` with the **Write tool, never argv**, `chmod 600`. Routine, not an
event. Mapping unchanged (4): `47913006`←`gpu-box-46996216`,
`49102182`←`gpu-box-31299601`, `50928407`←`gpu-box-46694776`,
`51415980`←`gpu-box-31947967`.

**CPU-BOUND rule unchanged:** decide against a control's first `stage2_step` `wall_s`
(`#478`, K=144/1024x16/batch256 → `wall_s 240`), **never a threshold**; exclude the k-fold
ridge solves and a run's first 1–2 h of anomaly transform + embed; **name the script**;
write the deadline and threshold down before the evidence closes; price both errors. Never
destroy from this session; never stop a box with a running job.

**For 22:29Z:** nothing in flight, no run to deadline, no box to watch. Append here rather
than creating a new file.

**No notification sent** — healthy, and nothing changed that Chris has not already been
told about.

---

## 2026-09-28 20:30Z — HEALTHY (exit 0), fleet unchanged from 19:30Z. 0 mutations, 0 fleet commits.

`fleet_health.mjs`: `1 run(s) not finished · 0 runner(s) online+idle` / HEALTHY / EXIT=0.
Two frames — `gpu_box.mjs list` ~20:31Z, then a direct `/api/v1/instances/` read ~20:32Z.

**Item 42 check: the schedule did NOT skip this hour.** The entry immediately below is
**19:30Z**, exactly one hour back. **Fifth consecutive clean hour** since the
13:30Z/14:30Z hole. Item 42 stays open (five hours is not a fix) and is **not
re-notified**.

**All five conditions clean; every Vast field is byte-identical to 19:30Z:**

- **CPU-BOUND** n/a — no box `running`, no job anywhere on the fleet, 0 of 21 runners
  online. No threshold used, no control read, no script to name.
- **IDLE BURN** clean — 4 instances, all `cur_state=stopped`/`intended=stopped`
  (`actual_status` exited/exited/loading/exited). **$0.00/h GPU burn.** No
  `gpu_box.mjs stop` issued.
- **QUEUE STALL** cannot fire — `#650 Test & Deploy`, queued 2026-08-19T04:04:09Z, is the
  known baseline (item 16); 0 runners online, so nothing to pair it with. It is now the
  **only** unfinished run: the count fell 2 → 1 because `#1240` and `#1241` both ended.
- **DISK** clean — `50928407` 132/200 = **66.0%** (highest), `51415980` 184/420 = 43.8%,
  `49102182` 89/300 = 29.7%, `47913006` 0.75/700 = 0.1%. All far under 90%.
- **TELEMETRY** clean — `gpu_temp` 60.0 / 38.0 / 49.0 / 53.0 °C, non-zero and identical
  across both frames. No dead frame. `50928407` still `stopped`/`loading`, the stale pair
  first noted 2026-09-25 12:30Z — sixth day, not billing GPU, not actioned.

**The hosted lanes closed as predicted, no finding.** `#1241 Test & Deploy`
(19:30Z's own commit `f7c6d47`, started 19:32:49Z) ended **failure** 20:09:19Z at ~37 min —
inside the expected band and the expected conclusion (standing red, `app.spec.js`). Its
start concurrency-**cancelled** `#1240` at 19:32:52Z, the same race logged at 10:30Z/11:30Z,
not a rule. Both on GitHub-**hosted** `ubuntu-latest`: no runner is online, so neither can
be a fleet job and neither costs Vast. `#221 tpu-status-mirror` **success** 18:20:01Z
unchanged; newest `slatrack` remains `#59` (14:33Z, failure, logged); newest `ml-train`
remains `#554`, 2026-09-07.

**Budget unchanged:** $0.00/h GPU; **$0.0815/h storage = $1.96/day** across the four
stopped boxes, re-summed from `storage_total_cost` (0.012963 + 0.011111 + 0.018519 +
0.038889) — identical to 19:30Z. Per 15:30Z, **do not sum `storage_cost`** — that field is
the host's $/GB/month rate. The ≈$41-of-$50 figure is still 03:30Z's **extrapolation, not a
reading**. Idle disk alone still reaches the $50 cap around **2026-10-02**; reported
08:30Z, per §0e never acted on from this session, **not re-notified.**

**Standing items.** **41 still open, BLOCKING** — the project store is still over its
2,000,000-token cap (660 docs listed this hour), so this file remains the record; nothing
deleted, that stays Chris's call. **42 open** (see above). 40, 37, 31, 16 unchanged.
15 — **110th** fresh-container bootstrap; re-cloned `earth` and rewrote `.gh_pat` +
`.vast_key` from `claude/github-access.md` Rule 2b + `claude/vast-access.md` with the
**Write tool, never argv**, `chmod 600`. Routine, not an event. Mapping unchanged (4):
`47913006`←`gpu-box-46996216`, `49102182`←`gpu-box-31299601`,
`50928407`←`gpu-box-46694776`, `51415980`←`gpu-box-31947967`.

**No notification sent** — healthy, and nothing changed that Chris has not already been
told about.

---

## 2026-09-28 19:30Z — HEALTHY (exit 0), fleet unchanged from 18:30Z. 0 mutations, 0 fleet commits.

`fleet_health.mjs`: `2 run(s) not finished · 0 runner(s) online+idle` / HEALTHY / EXIT=0.
Two frames — `gpu_box.mjs list` ~19:31Z, then a direct `/api/v1/instances/` read ~19:32Z.

**Item 42 check: the schedule did NOT skip this hour.** The entry immediately below is
**18:30Z**, exactly one hour back. **Fourth consecutive clean hour** since the
13:30Z/14:30Z hole. Item 42 stays open (four hours is not a fix) and is **not
re-notified**.

**All five conditions clean; every Vast field is byte-identical to 18:30Z:**

- **CPU-BOUND** n/a — no box `running`, no job anywhere on the fleet, 0 of 21 runners
  online. No threshold used, no control read, no script to name.
- **IDLE BURN** clean — 4 instances, all `cur_state=stopped`/`intended=stopped`
  (`actual_status` exited/exited/loading/exited). **$0.00/h GPU burn.** No
  `gpu_box.mjs stop` issued.
- **QUEUE STALL** cannot fire — `#650 Test & Deploy`, queued 2026-08-19T04:04:09Z, is the
  known baseline (item 16); 0 runners online, so nothing to pair it with.
- **DISK** clean — `50928407` 132/200 = **66.0%** (highest), `51415980` 184/420 = 43.8%,
  `49102182` 89/300 = 29.7%, `47913006` 0.8/700 = 0.1%. All far under 90%.
- **TELEMETRY** clean — `gpu_temp` 60.0 / 38.0 / 49.0 / 53.0 °C, non-zero and identical
  across both frames. No dead frame. `50928407` still `stopped`/`loading`, the stale pair
  first noted 2026-09-25 12:30Z — sixth day, not billing GPU, not actioned.

**The second unfinished run is the predicted one, not a finding.** `#1240 Test & Deploy`
(`36465992996`, 18:30Z's own commit `ea2d3d8`, started 18:33:13Z) is `in_progress` at
~57 min on a GitHub-**hosted** `ubuntu-latest` — exactly what 18:30Z predicted (`~39–60
min`, expected `failure`). It cannot be a fleet job: no runner is online. `#1239` already
ended `failure` 18:32:36Z and was logged last hour. Also routine: `#221
tpu-status-mirror` **success** 18:20:01Z, unchanged. Newest `slatrack` remains `#59`
(14:33Z, failure, logged); newest `ml-train` remains `#554`, 2026-09-07. Hosted lanes, no
Vast cost, outside the five conditions.

**Budget unchanged:** $0.00/h GPU; **$0.0815/h storage = $1.96/day** across the four
stopped boxes, re-summed from `storage_total_cost` (0.012963 + 0.011111 + 0.018519 +
0.038889). Per 15:30Z, **do not sum `storage_cost`** — that field is the host's $/GB/month
rate. The ≈$41-of-$50 figure is still 03:30Z's **extrapolation, not a reading**. Idle disk
alone still reaches the $50 cap around **2026-10-02**; reported 08:30Z, per §0e never
acted on from this session, **not re-notified.**

**Standing items.** **41 still open, BLOCKING** — project store over its 2,000,000-token
cap; `project_write` re-attempted this hour and still refused, so this file remains the
record; nothing deleted, that stays Chris's call. **42 open** (see above). 40, 37, 31, 16
unchanged. 15 — **109th** fresh-container bootstrap; re-cloned `earth` and rewrote
`.gh_pat` + `.vast_key` from `claude/github-access.md` Rule 2b + `claude/vast-access.md`
with the **Write tool, never argv**, `chmod 600`. Routine, not an event. Mapping unchanged
(4): `47913006`←`gpu-box-46996216`, `49102182`←`gpu-box-31299601`,
`50928407`←`gpu-box-46694776`, `51415980`←`gpu-box-31947967`.

**CPU-BOUND rule unchanged:** decide against a control's first `stage2_step` `wall_s`
(`#478`, K=144/1024x16/batch256 → `wall_s 240`), **never a threshold**; exclude the k-fold
ridge solves and a run's first 1–2 h of anomaly transform + embed; **name the script**;
write the deadline and threshold down before the evidence closes; price both errors. Never
destroy from this session; never stop a box with a running job.

**No notification sent this hour** — healthy, and every reading unchanged from 18:30Z.

**For 20:30Z:** nothing in flight, no run to deadline, no box to watch. This commit will
trigger `#1241`, expected `failure` in ~39–60 min — not a finding either way. **Check the
entry below yours: if it is not 19:30Z, the schedule skipped again, and that is
reportable.** Append here rather than creating a new file.

---

## 2026-09-28 18:30Z — HEALTHY (exit 0), fleet unchanged from 17:30Z. 0 mutations, 0 fleet commits.

`fleet_health.mjs`: `2 run(s) not finished · 0 runner(s) online+idle` / HEALTHY / EXIT=0.
Two frames — `gpu_box.mjs list` ~18:32Z, then a direct `/api/v1/instances/` read ~18:33Z.

**Item 42 check: the schedule did NOT skip this hour.** The entry immediately above is
**17:30Z**, exactly one hour back. **Third consecutive clean hour** since the 13:30Z/14:30Z
hole. Item 42 stays open (three hours is not a fix) and is **not re-notified**.

**All five conditions clean; every Vast field is byte-identical to 17:30Z:**

- **CPU-BOUND** n/a — no box `running`, no job anywhere on the fleet, 0 of 21 runners
  online. No threshold used, no control read, no script to name.
- **IDLE BURN** clean — 4 instances, all `cur_state=stopped`/`intended=stopped`
  (`actual_status` exited/exited/loading/exited). **$0.00/h GPU burn.** No
  `gpu_box.mjs stop` issued.
- **QUEUE STALL** cannot fire — `#650 Test & Deploy`, queued 2026-08-19T04:04:09Z, is the
  known baseline (item 16); 0 runners online, so nothing to pair it with.
- **DISK** clean — `50928407` 132/200 = **66.0%** (highest), `51415980` 184/420 = 43.8%,
  `49102182` 89/300 = 29.7%, `47913006` 0.8/700 = 0.1%. All far under 90%.
- **TELEMETRY** clean — `gpu_temp` 60.0 / 38.0 / 49.0 / 53.0 °C, non-zero and identical
  across both frames. No dead frame. `50928407` still `stopped`/`loading`, the stale pair
  first noted 2026-09-25 12:30Z — sixth day, not billing GPU, not actioned.

**The second unfinished run is the predicted one, not a finding.** `#1239 Test & Deploy`
(17:30Z's own commit `292b36e`, started 17:33:08Z) is `in_progress` at ~60 min on a
GitHub-**hosted** `ubuntu-latest` — exactly what 17:30Z predicted (`~39–60 min`, expected
`failure`). It cannot be a fleet job: no runner is online. `#1238` already failed at
17:31:23Z and was logged last hour. Also routine this hour: `#221 tpu-status-mirror`
**success** 18:20:01Z (21 s). Newest `slatrack` remains `#59` (14:33Z, failure, logged);
newest `ml-train` remains `#554`, 2026-09-07. Hosted lanes, no Vast cost, outside the five
conditions.

**Budget unchanged:** $0.00/h GPU; **$0.0815/h storage = $1.96/day** across the four
stopped boxes, re-summed from `storage_total_cost` (0.012963 + 0.011111 + 0.018519 +
0.038889). Per 15:30Z, **do not sum `storage_cost`** — that field is the host's $/GB/month
rate. The ≈$41-of-$50 figure is still 03:30Z's **extrapolation, not a reading**. Idle disk
alone still reaches the $50 cap around **2026-10-02**; reported 08:30Z, per §0e never acted
on from this session, **not re-notified.**

**Standing items.** **41 still open, BLOCKING** — project store over its 2,000,000-token
cap; `project_write` re-attempted this hour and still refused, so this file remains the
record; nothing deleted, that stays Chris's call. **42 open** (see above). 40, 37, 31, 16
unchanged. 15 — **108th** fresh-container bootstrap; re-cloned `earth` and rewrote
`.gh_pat` + `.vast_key` from `claude/github-access.md` Rule 2b + `claude/vast-access.md`
with the **Write tool, never argv**, `chmod 600`. Routine, not an event. Mapping unchanged
(4): `47913006`←`gpu-box-46996216`, `49102182`←`gpu-box-31299601`,
`50928407`←`gpu-box-46694776`, `51415980`←`gpu-box-31947967`.

**CPU-BOUND rule unchanged:** decide against a control's first `stage2_step` `wall_s`
(`#478`, K=144/1024x16/batch256 → `wall_s 240`), **never a threshold**; exclude the k-fold
ridge solves and a run's first 1–2 h of anomaly transform + embed; **name the script**;
write the deadline and threshold down before the evidence closes; price both errors. Never
destroy from this session; never stop a box with a running job.

**No notification sent this hour** — healthy, and every reading unchanged from 17:30Z.

**For 19:30Z:** nothing in flight, no run to deadline, no box to watch. This commit will
trigger `#1240`, expected `failure` in ~39–60 min — not a finding either way. **Check the
entry above this one: if it is not 18:30Z, the schedule skipped again, and that is
reportable.** Append here rather than creating a new file.

---

## 2026-09-28 17:30Z — HEALTHY (exit 0), fleet unchanged from 16:30Z. 0 mutations, 0 fleet commits.

`fleet_health.mjs`: `1 run(s) not finished · 0 runner(s) online+idle` / HEALTHY / EXIT=0.
Two frames — `gpu_box.mjs list` ~17:31Z, then a direct `/api/v1/instances/` read ~17:32Z.

**Item 42 check: the schedule did NOT skip this hour either.** The entry immediately above
is **16:30Z**, exactly one hour back. Second consecutive clean hour since the 13:30Z/14:30Z
hole. Item 42 stays open (two hours is not a fix) and is **not re-notified**.

**All five conditions clean; every Vast field is byte-identical to 16:30Z:**

- **CPU-BOUND** n/a — no box `running`, no job anywhere on the fleet. No threshold used,
  no control read, no script to name.
- **IDLE BURN** clean — 4 instances, all `cur_state=stopped`/`intended=stopped`, 0 of 21
  runners online. **$0.00/h GPU burn.** No `gpu_box.mjs stop` issued.
- **QUEUE STALL** cannot fire — `#650 Test & Deploy`, queued 2026-08-19T04:04:09Z, is the
  known baseline (item 16) and is the whole of the `1 run not finished`; 0 runners online,
  so nothing to pair it with. Every one of the last 25 runs is `completed`.
- **DISK** clean — `50928407` 132/200 = **66.0%** (highest), `51415980` 184/420 = 43.8%,
  `49102182` 89/300 = 29.7%, `47913006` 0.8/700 = 0.1%. All far under 90%.
- **TELEMETRY** clean — `gpu_temp` 60.0 / 38.0 / 49.0 / 53.0 °C, non-zero and identical
  across both frames. No dead frame. `50928407` still `stopped`/`loading`, the stale pair
  first noted 2026-09-25 12:30Z — fifth day, not billing GPU, not actioned.

**Hosted workflows — one verdict, predicted, not reportable.** `#1238 Test & Deploy`
(16:30Z's own commit `f18617e`, started 16:32:00Z) **failed at 17:31:23Z after 59.4 min**,
which is the trajectory 16:30Z predicted; the band ticks 38.9–58.8 → **38.9–59.4**.
**Eleventh straight non-cancelled `failure`** — red for days, still not actioned, still not
new. No new `slatrack` run since `#59` (14:33Z, already logged); newest `ml-train` remains
#554, 2026-09-07. Hosted `ubuntu-latest`, no Vast cost, outside the five conditions.

**Budget unchanged:** $0.00/h GPU; **$0.0815/h storage = $1.96/day** across the four
stopped boxes, re-summed from `storage_total_cost` (0.012963 + 0.011111 + 0.018519 +
0.038889). Per 15:30Z, **do not sum `storage_cost`** — that field is the host's $/GB/month
rate. The ≈$41-of-$50 figure is still 03:30Z's **extrapolation, not a reading**. Idle disk
alone still reaches the $50 cap around **2026-10-02**; reported 08:30Z, per §0e never acted
on from this session, **not re-notified.**

**Standing items.** **41 still open, BLOCKING** — project store over its 2,000,000-token
cap, `project_write` still refused, so this file remains the record; nothing deleted, that
stays Chris's call. **42 open** (see above). 40, 37, 31, 16 unchanged. 15 — **107th**
fresh-container bootstrap; re-cloned `earth` and rewrote `.gh_pat` + `.vast_key` from
`claude/github-access.md` Rule 2b + `claude/vast-access.md` with the **Write tool, never
argv**, `chmod 600`. Routine, not an event. Mapping unchanged (4):
`47913006`←`gpu-box-46996216`, `49102182`←`gpu-box-31299601`,
`50928407`←`gpu-box-46694776`, `51415980`←`gpu-box-31947967`.

**CPU-BOUND rule unchanged:** decide against a control's first `stage2_step` `wall_s`
(`#478`, K=144/1024x16/batch256 → `wall_s 240`), **never a threshold**; exclude the k-fold
ridge solves and a run's first 1–2 h of anomaly transform + embed; **name the script**;
write the deadline and threshold down before the evidence closes; price both errors. Never
destroy from this session; never stop a box with a running job.

**No notification sent this hour** — healthy, and every reading unchanged from 16:30Z.

**For 18:30Z:** nothing in flight, no run to deadline, no box to watch. This commit will
trigger `#1239`, expected `failure` in ~39–60 min — not a finding either way. **Check the
entry above this one: if it is not 17:30Z, the schedule skipped again, and that is
reportable.** Append here rather than creating a new file.

---

## 2026-09-28 16:30Z — HEALTHY (exit 0), fleet unchanged from 15:30Z. 0 mutations, 0 fleet commits.

`fleet_health.mjs`: `1 run(s) not finished · 0 runner(s) online+idle` / HEALTHY / EXIT=0.
Two frames — `gpu_box.mjs list` ~16:31Z, then a direct `/api/v1/instances/` read ~16:32Z.

**Item 42's own check, first application: the schedule did NOT skip this hour.** The entry
immediately above is **15:30Z**, exactly one hour back, so the 13:30Z/14:30Z hole did not
repeat. Item 42 stays open (one clean hour is not a fix) but is **not re-notified**.

**All five conditions clean; every Vast field is byte-identical to 15:30Z:**

- **CPU-BOUND** n/a — no box `running`, no job anywhere on the fleet. No threshold used,
  no control read, no script to name.
- **IDLE BURN** clean — 4 instances, all `cur_state=stopped`/`intended=stopped`, 0 of 21
  runners online. **$0.00/h GPU burn.** No `gpu_box.mjs stop` issued.
- **QUEUE STALL** cannot fire — `#650 Test & Deploy`, queued 2026-08-19T04:04:09Z, is the
  known baseline (item 16) and is the whole of the `1 run not finished` count; 0 runners
  online, so nothing to pair it with.
- **DISK** clean — `50928407` 132/200 = **66.0%** (highest), `51415980` 184/420 = 43.8%,
  `49102182` 89/300 = 29.7%, `47913006` 0.8/700 = 0.1%. All far under 90%.
- **TELEMETRY** clean — `gpu_temp` 60.0 / 38.0 / 49.0 / 53.0 °C, non-zero and identical
  across both frames. No dead frame. `50928407` still `stopped`/`loading`, the stale pair
  first noted 2026-09-25 12:30Z — fifth day, not billing GPU, not actioned.

**Hosted workflows — nothing new and nothing reportable.** `#1237 Test & Deploy`
(15:30Z's own commit `cb4b843`, started 15:35:55Z) is `in_progress` at ~55 min, which is
the predicted trajectory, not a hang — it is the *only* unfinished run in the last 30.
`#59 slatrack fetch` and `#1236` were already logged as verdicts at 15:30Z; no further
`slatrack` or `ml-train` run has started since (newest `ml-train` remains #554, 2026-09-07).
Everything else green in the window. Hosted `ubuntu-latest`, no Vast cost, outside the
five conditions.

**Budget unchanged:** $0.00/h GPU; **$0.0815/h storage = $1.96/day** across the four
stopped boxes, re-summed from `storage_total_cost` (0.012963 + 0.011111 + 0.018519 +
0.038889). Per 15:30Z, **do not sum `storage_cost`** — that field is the host's
$/GB/month rate. The ≈$41-of-$50 figure is still 03:30Z's **extrapolation, not a reading**.
Idle disk alone still reaches the $50 cap around **2026-10-02**; reported 08:30Z, per §0e
never acted on from this session, **not re-notified.**

**Standing items.** **41 still open, BLOCKING** — project store over its 2,000,000-token
cap, every `project_write` still refused, so this file remains the record; nothing deleted,
that stays Chris's call. **42 open** (see above). 40, 37, 31, 16 unchanged. 15 — **106th**
fresh-container bootstrap; re-cloned `earth` and rewrote `.gh_pat` + `.vast_key` from
`claude/github-access.md` Rule 2b + `claude/vast-access.md` with the **Write tool, never
argv**, `chmod 600`. Routine, not an event. Mapping unchanged (4):
`47913006`←`gpu-box-46996216`, `49102182`←`gpu-box-31299601`,
`50928407`←`gpu-box-46694776`, `51415980`←`gpu-box-31947967`.

**CPU-BOUND rule unchanged:** decide against a control's first `stage2_step` `wall_s`
(`#478`, K=144/1024x16/batch256 → `wall_s 240`), **never a threshold**; exclude the k-fold
ridge solves and a run's first 1–2 h of anomaly transform + embed; **name the script**;
write the deadline and threshold down before the evidence closes; price both errors. Never
destroy from this session; never stop a box with a running job.

**For 17:30Z:** nothing in flight, no run to deadline, no box to watch. This commit will
trigger `#1238`, expected `failure` in ~39–59 min — not a finding either way. **Check the
entry above this one: if it is not 16:30Z, the schedule skipped again, and that is
reportable.** Append here rather than creating a new file.

---

## 2026-09-28 15:30Z — HEALTHY (exit 0), fleet unchanged from 12:30Z. 0 mutations. **The 13:30Z and 14:30Z checks never ran.**

`fleet_health.mjs`: `1 run(s) not finished · 0 runner(s) online+idle` / HEALTHY / EXIT=0.
Two frames — `gpu_box.mjs list` ~15:32Z, then a direct `/api/v1/instances/` read ~15:35Z.

**The finding this hour is not on the fleet, it is in this routine.** The last entry above
is 12:30Z and the last commit on `main` is its own, `eaed832` at 12:34:19Z. A
`GET /commits?sha=main&since=2026-09-28T12:00:00Z` returns **that one commit and nothing
else**, and there is no 13:30Z or 14:30Z entry in this file. Every check commits its entry,
so **two consecutive hourly checks did not run** — the watch was dark from 12:34Z to
15:30Z, just under three hours. Nothing was at risk during the hole (no box `running`, no
job dispatched, $0.00/h GPU), which is why it cost nothing this time; it would have
mattered during a training run. **Chris notified.** Not something this session can fix from
inside — the schedule lives outside the repo.

**All five conditions clean, and every fleet reading is byte-identical to 12:30Z:**

- **CPU-BOUND** n/a — no box `running`, no job anywhere on the fleet. No threshold used,
  no control read, no script to name.
- **IDLE BURN** clean — 4 instances, all `cur_state=stopped`/`intended=stopped`, 0 of 21
  runners online. **$0.00/h GPU burn.** No `gpu_box.mjs stop` issued.
- **QUEUE STALL** cannot fire — `#650 Test & Deploy`, queued 2026-08-19T04:04:09Z, is the
  known baseline (item 16) and is the whole of the `1 run not finished`; 0 runners online,
  so nothing to pair it with. No run of the last 30 is unfinished.
- **DISK** clean — `50928407` 132/200 = **66.0%** (highest), `51415980` 184/420 = 43.8%,
  `49102182` 89/300 = 29.7%, `47913006` 0.75/700 = 0.1%. Unchanged, all far under 90%.
- **TELEMETRY** clean — `gpu_temp` 60.0 / 38.0 / 49.0 / 53.0 °C, all non-zero, identical
  across both frames, no dead frame. `50928407` still `stopped`/`loading`, the stale pair
  first noted 2026-09-25 12:30Z — fourth day, not billing GPU, not actioned.

**Two hosted-workflow verdicts arrived, neither a fleet condition and neither new.**
`#1236 Test & Deploy` (12:30Z's own commit `eaed832`) **failed at 13:33:15Z after 58.8
min** — inside the band, which ticks 38.9–58.5 → **38.9–58.8**. Tenth straight
non-cancelled `failure`; red for days, still not actioned, still not new.
`#59 slatrack fetch 1993-2024` **failed 14:33:14Z**, one lane of six — `fetch (2019-2024)`
at step 8, *"Every year of the lane must carry a done.json"*; the other five lanes green.
That is the **known intermittency** the 03:30Z baseline recorded (#55, #56 failed 09-27,
#57/#58 green) — now 4 failures in the last 12 scheduled runs. Hosted `ubuntu-latest`, no
Vast cost, outside the five conditions. Everything else green: `#98 GLORYS pull (global)`
14:07Z, `#12 Daily loitering refresh` 13:46Z, `#166 GLORYS pull` 13:08Z. No new `ml-train`
run — newest remains #554, 2026-09-07.

**Budget — unchanged, and a field trap worth recording.** $0.00/h GPU; **$0.0815/h storage
= $1.96/day** across the four stopped boxes, re-summed from `storage_total_cost`
(0.012963 + 0.011111 + 0.018519 + 0.038889). **Do not sum `storage_cost`** — that field is
the host's $/GB/month rate (0.0133 / 0.0267 / 0.0667 / 0.0667 here) and summing it reads
$0.173/h, more than double the truth. Caught and discarded this hour before it became a
finding. `storage_total_cost` is the per-hour charge. The ≈$41-of-$50 figure remains
03:30Z's **extrapolation, not a reading** — `/users/current/` exposes no balance field, and
`credit_balance` is `null` on every instance. At $1.96/day with nothing training, idle disk
alone reaches the $50 cap around **2026-10-02**, buying no science; reported 08:30Z, per
§0e never acted on from this session.

**Standing items.** **41 still open, BLOCKING** — the project store is over its
2,000,000-token cap; every `project_write` still fails, so this file remains the record.
~610 superseded hourly `fleet-check-*.md` docs are the bulk; deleting project docs is
outside this routine's remit, so **nothing was deleted.** Sent 08:30Z, not re-notified.
40, 31, 37, 16 unchanged. 15 — **105th** fresh-container bootstrap; re-cloned `earth` and
rewrote `.gh_pat` + `.vast_key` from `claude/github-access.md` Rule 2b +
`claude/vast-access.md` with the **Write tool, never argv**, `chmod 600`. Routine, not an
event. Mapping unchanged (4): `47913006`←`gpu-box-46996216`, `49102182`←`gpu-box-31299601`,
`50928407`←`gpu-box-46694776`, `51415980`←`gpu-box-31947967`.
**42 NEW (open, needs Chris):** the hourly schedule silently skipped 13:30Z and 14:30Z.
A missed check leaves no trace anywhere except the hole in this file, so **a future check
should compare its own hour against the entry above it** and say so when the gap is >1 h.

**CPU-BOUND rule unchanged:** decide against a control's first `stage2_step` `wall_s`
(`#478`, K=144/1024x16/batch256 → `wall_s 240`), **never a threshold**; exclude the k-fold
ridge solves and a run's first 1–2 h of anomaly transform + embed; **name the script**;
write the deadline and threshold down before the evidence closes; price both errors. Never
destroy from this session; never stop a box with a running job.

**For 16:30Z:** nothing in flight, no run to deadline, no box to watch. This commit will
trigger `#1237`, expected `failure` in ~39–59 min — not a finding either way. **Check the
entry above this one for its hour: if it is not 15:30Z, the schedule skipped again, and
that is reportable.** Append here rather than creating a new file.

---

## 2026-09-28 12:30Z — HEALTHY (exit 0), fleet unchanged from 11:30Z. 0 mutations, 0 fleet commits.

`fleet_health.mjs`: `1 run(s) not finished · 0 runner(s) online+idle` / HEALTHY / EXIT=0.
Two frames — `gpu_box.mjs list` then a direct `/api/v1/instances/` read, ~12:33–12:35Z.

- **CPU-BOUND** n/a — no box `running`, no job anywhere on the fleet. No threshold used,
  no control read, no script to name.
- **IDLE BURN** clean — 4 instances, all `cur_state=stopped`/`intended=stopped`, 0 of 21
  runners online. **$0.00/h GPU burn.** No `gpu_box.mjs stop` issued.
- **QUEUE STALL** cannot fire — `#650 Test & Deploy`, queued 2026-08-19T04:04:09Z, is the
  known baseline (item 16); 0 runners online, so nothing to pair it with.
- **DISK** clean — `50928407` 132/200 = **66.0%** (highest), `51415980` 184/420 = 43.8%,
  `49102182` 89/300 = 29.7%, `47913006` 0.8/700 = 0.1%. Byte-identical to 11:30Z, all far
  under 90%.
- **TELEMETRY** clean — `gpu_temp` 60.0 / 38.0 / 49.0 / 53.0 °C, all non-zero, no dead
  frame. `50928407` still `stopped`/`loading`, the stale pair first noted 2026-09-25
  12:30Z — fourth day, not billing GPU, not actioned.

**Nothing changed on the fleet this hour, so Chris was not notified.**

**One data point that widens 11:30Z's band, and nothing more.** `#1235 Test & Deploy`
(11:30Z's own commit `50d155e`) reached its own verdict — **`failure` at 12:11:18Z after
38.9 min**, well short of the 46.6–58.5 min band the last nine non-cancelled runs sat in.
Same red workflow, sooner. So the band is **46.6–58.5 min → now 38.9–58.5**, and the
race 11:30Z described still holds: a run that finishes in ~39 min clears comfortably
before the next hourly commit, which is why the count reads **1** again rather than 2.
Hosted `ubuntu-latest`, no Vast cost, outside the five conditions — noted, not actioned,
not new. Everything else green: `#71 Daily forecast refresh` 11:56Z, `#220
tpu-status-mirror` 10:51Z, `#58 slatrack fetch` 05:58Z, `#97`/`#165 GLORYS` 05:33Z/04:08Z.
No new `ml-train` run — newest remains #554, 2026-09-07. HEAD `764e1bc` (automated GFS
refresh, 11:54Z).

**Budget — unchanged.** $0.00/h GPU; **$0.0815/h storage = $1.96/day** across the four
stopped boxes (0.01296 + 0.01111 + 0.01852 + 0.03889, re-summed this hour). The
≈$41-of-$50 figure remains 03:30Z's **extrapolation, not a reading** — Vast
`/users/current/` exposes no credit or balance field. At $1.96/day with nothing training,
idle disk alone reaches the $50 cap around **2026-10-02**, buying no science. Reported at
08:30Z; per §0e never acted on — no box stopped, started or destroyed from this session.

**Standing items.** **41 still open, BLOCKING, needs Chris** — the project store is over
its 2,000,000-token cap; every project write still fails for every session. ~610
superseded hourly `fleet-check-*.md` docs are the bulk; deleting project docs is outside
this routine's remit, so **nothing was deleted.** Sent at 08:30Z, unchanged, not
re-notified. 15 — 102nd fresh-container bootstrap; re-cloned `earth` and rewrote
`.gh_pat` + `.vast_key` from `claude/github-access.md` Rule 2b + `claude/vast-access.md`
with the **Write tool, never argv**, `chmod 600`. Routine, not an event. 40, 31, 37
unchanged. Mapping unchanged (4): `47913006`←`gpu-box-46996216`,
`49102182`←`gpu-box-31299601`, `50928407`←`gpu-box-46694776`, `51415980`←`gpu-box-31947967`.

**CPU-BOUND rule unchanged:** decide against a control's first `stage2_step` `wall_s`
(`#478`, K=144/1024x16/batch256 → `wall_s 240`), **never a threshold**; exclude the k-fold
ridge solves and a run's first 1–2 h of anomaly transform + embed; **name the script**;
write the deadline and threshold down before the evidence closes; price both errors. Never
destroy from this session; never stop a box with a running job.

**For 13:30Z:** nothing in flight, no run to deadline, no box to watch. This commit will
trigger `#1236`; whether 13:30Z sees it `failure` or `cancelled` turns on the race above —
either way it is not a finding. Append here rather than creating a new file.

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

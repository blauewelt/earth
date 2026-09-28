# Rolling fleet-check log

One compact entry per hourly check. **Deliberately one file, not one file per
hour** — ~610 per-hour `fleet-check-*.md` docs are what pushed the project store
over its 2,000,000-token cap (standing item 41). Do not reintroduce that pattern
here. Long-form records only when an hour actually warrants one; the
2026-09-28 03:30Z doc is the current baseline and stays as its own file.

Newest first.

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

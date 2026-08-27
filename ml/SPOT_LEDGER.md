# The spot-TPU ledger — issues and extra costs, appended as they happen

**The standing rule (Chris, 2026-08-27): every TPU dispatch tries SPOT
first**, in the launch zone, then across the alternative-zone ladder
(us-west1-c · us-west4-a/b · us-central1-a · us-east5-a/b/c); on-demand is
the fallback, never the first ask. The machinery that makes this safe is in
place and exercised: checkpoints ship to the bucket every `CKPT_EVERY` steps,
a relaunch under the same node name resumes exactly (optimizer state and
schedule position included), the boot beacon gives a ~6-minute zombie verdict,
and `tpu_box.py`'s lemon guard refuses born-unhealthy hosts before they bill
more than pennies.

**This file is the honest price of that rule.** Every spot-specific incident
— a preemption, a lemon host, a zone with no capacity, a run that had to fall
back to on-demand — gets a row here, with its measured cost, *in the same
session that hits it* (the same discipline as `ml/EXPERIMENTS.md`: the entry
is part of the work, not an afterthought). Costs use the day's assumption of
**$2.40/h per spot v5litepod-4** (50% of the $4.80 list) unless the console
rate was actually read; node-hours come from the TPU operations ledger and
are exact.

## How to read the error codes (all measured, not from docs)

| what GCP says | what it means |
|---|---|
| `429 Quota limit ... exceeded. Limit: N` | FIRST check the cores you asked for against N — a v5litepod-8 against a 4-core grant produces this in every zone with nothing billing (2026-08-26, ~2 h lost). Only then is it a real quota ceiling. |
| op error `code 5 Reservation not found` | SPOT CAPACITY refusal in disguise (measured 08-22 and 08-27 in us-east5) — not a config error, nothing to fix; try another zone. |
| op error `code 8 no more capacity / insufficient capacity` | plain capacity; spot AND on-demand can both be dry (us-central1-a, 08-27). |
| node READY but `runs/<exp>/` empty ≥6 min after boot | the startup script never ran (a zombie) — delete and redraw; the beacon makes this verdict certain. |
| health note "The TPU had a maintenance event ..." | decides NOTHING by itself — healthy on-demand nodes carry it too (08-27); only the beacon separates zombie from fine. |

## Incident log

| date (UTC) | incident | extra cost |
|---|---|---|
| 08-22 | First provisioning day: spot capacity existed in exactly ONE zone (us-west1-c); us-central1-a dry both kinds, us-east5-a/c refused spot as `Reservation not found`. | zone-shopping time only |
| 08-22 | The §8 orphan: a node outlived its session before the self-reap machinery existed — 8.1 h idle. (Not a spot failure per se; the lesson that produced the EXIT-trap + watchdog + hard-cap stack every launcher now carries.) | ~$19–39 (kind unrecorded) |
| 08-26 | **Lemon epidemic in us-west1-c spot**: the pool repeatedly served hosts born `UNHEALTHY_MAINTENANCE` that never ran their startup script — ≥5 sightings across two waves (09:20Z, 12:19Z) plus refusals at 20:02Z and 23:04Z. Produced the hardcoded `LEMON_HOSTS` + born-unhealthy guard in tpu_box.py (Chris: "hardcode somewhere that the buggy node will never get used again"). | ~$2–3 of short-lived nodes |
| 08-26 | Diagnosis noise, not spot's fault, recorded to keep the ledger honest: the 16:18Z and 18:56Z spot nodes were misread as pool zombies but were actually running blind on an incompletely-baked script (`__BUCKET__` unsubstituted). Spot was exonerated; the placeholder guard and boot beacon came out of it. | ~$3 (bake bug, not spot) |
| 08-27 | E-051-continuation start (other session): two sub-8-minute spot nodes before the third stuck (09:41Z, 09:51Z) — consistent with lemon redraws; the third has run cleanly since 10:02Z. | ~$0.6 |
| 08-27 | us-central1-a: NO capacity for v5litepod-4, spot AND on-demand (code 8). E-052.1b zone-shopped to us-west4-a, where **spot served a healthy node first try**. | ~15 min |
| 08-27 14:31Z | **Arm-A migration blocked — the day's spot capacity was exactly the two slots already in use.** Tried to move e052-1-train (on-demand, ~12.5 h left) to spot: us-east5-a/c `Reservation not found`, us-east5-b + us-west4-b `insufficient capacity`. The run finishes on-demand; this is the ledger's first entry in the "could not launch spot at all" category. | **≈$60** on-demand premium accepted |
| — | **Preemptions observed so far: none.** (When one lands: date, node, step lost since last shipped ckpt, relaunch gap, and any repeated-preemption churn go here.) | — |

## Running totals (update with each entry)

- Extra cost attributable to spot itself (lemons, redraws): **≈$6** to date.
- On-demand premium paid because spot was unavailable: **≈$60** to date
  (arm A, 08-27) — plus the E-051 main run and everything before 08-26,
  which predate the spot-first rule and are not counted against it.
- Savings from runs that did land on spot (vs list): **≈$50+** to date and
  growing (arm B ≈$48/run, E-051 continuation, verify nodes).

The comparison to keep in view: one lemon redraw costs cents and six minutes;
one 22-hour training run on spot saves ≈$50. The rule pays for itself unless
preemption churn (not yet observed) changes the arithmetic — which is exactly
what this file exists to notice.

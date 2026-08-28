# The standing overview — every experiment, one line each, and what's next

**Last updated: 2026-08-28 ~14:05 UTC** (Opus). **E-054b is now running the
RIGHT experiment.** Its 07:08Z relaunch had been rebuilt from the pristine
launcher template with only the width/steps/tag sed'd in, so it silently
reverted every E-051 knob (Z_ASSET empty, K 24, lr 1e-3, stencil 1, znoise 0,
grad_clip 0) and spent an hour re-embedding an already-published Z; caught by
reading the `resolved knobs` line, killed 08:07:55Z, and relaunched 08:10:33Z
on us-west1-c spot first try with the full E-054a knob block `diff`-verified
against E-054a's own startup file. It now reads `K 144 · lr 4e-4 · 1280x20 ·
stencil 145 · znoise 0.7 · grad_clip 128 · grad_accum 4 (micro 64)` and
**pulled** the 16.24 GiB Z instead of rebuilding it. Gradient accumulation is
certified exact (max rel 2.4e-07), so this is a memory decomposition of the
same batch-256 step, not a batch change. **E-058 rung 1 of Chris's
multi-target directive is BUILT and certified with no GPU spent**: rolled
skill is now decomposed per channel, so the next roll answers "does this
predict SST?" — pooled numbers bit-identical (18,289 bytes), consistency
1.11e-16, and the two byte-identity tripwires widened under their own
strip-count-pin pattern rather than relaxed. **#504 (E-056a-R) is at step
6000/20000, ratio 0.574** (0.36416/0.63451) — improving from 0.600 at 2800,
still above the continuous control 0.5056 and well above the 0.4394 lattice
bar. **E-056 IS RESOLVED (one-step): the token road does NOT open.** #507, the
dose-matched twin, finished at **0.50986** against #504's 0.53873 — same box,
same codec, same 20k steps, only the noise differs, and the dose-matching is
confirmed exact (`rel_pers` 0.15065 vs the controls' 0.15116). So the
pre-registered confound was real and worth an arm: it bought **-0.029**, an
order of magnitude above the 0.00113 pair spread at this tier. But 0.51 is
still on the wrong side of the registered ">=0.50 lost the signal" line: it is
at PARITY with the continuous d_z-32 control (0.5056, +0.0043 — a consistency,
not a gap) and **0.070 behind the lattice d_z-32 bar (0.4394)** that the wave
was registered on. A 16-bit alphabet at d_z 6 carries what continuous d_z-32
carries at ~5% of the size — a real efficiency finding — but ≲0.44 was the bar
and 0.51 does not clear it. **E-056b is DROPPED**, not re-dispatched: refining
a denser slab within a substrate that failed at K=24 would spend ~$20 on a
closed road. The next full-budget pentad head stays on continuous z; E-046's
train-THROUGH-the-lattice result is the survivor worth pursuing. **#504 (token substrate) is COMPLETE, and its probe rows are NOT subject to the
dose confound.** One-step ratio **0.53373** (0.33292/0.62375, self-consistent
within `temporal.json`) against continuous d_z-32 0.5056 and lattice 0.4394 —
worse than both. That row IS confounded (5.8x the intended relative dose;
#507 decides it). But `--input-znoise` is a STAGE-2 knob and the probes score
the FROZEN CODEC, so the probe rows are clean, and they point the same way:
unpooled head on RAPID **0.588** against its own raw-3x3 control **0.693** and
the unpooled wind bar **0.690**, and Florida Current **0.051** (CI contains
zero) against a 0.199 wind bar. Two independent instruments, one unconfounded
— but §3b's head-probe seed regime is 0.036-0.245 and every pentad arm is
n = 1, so this is a direction, not a level. **#507** (dose-matched, znoise
0.12) started 11:24Z on the same box and is healthy at **0.237 s/step, gpu
99.99%** — done ~13:10Z plus its ladder. **#506 (K=144) was CANCELLED at 12:30Z: the rented H100 never used its GPU.**
Ten `gpu_util` samples across 3 h 55 m all read 0% at `cpu_util` 95-97% (real
readings — the dead-frame check never fired), and it wrote ZERO step records
in 50 minutes where its own control #478, identical geometry, writes its first
at `wall_s 240`. The pre-registered 12:30Z rule was held to rather than
improvised. Cost ~$7.95 against the ~$20 the remaining timeout would have
spent; nothing was lost with it. **E-056b is HELD, not re-dispatched** — its
question (does the dense slab help WITHIN tokens?) only matters if #507 says
the substrate survives at the dose-matched noise, which it answers in 2-3 h.
**E-054b's first training
step at 08:37Z showed NO OOM** — the registered HBM risk is closed by gradient
accumulation, and its `val_persistence 21.44621` is bit-identical to E-054a's,
so the ratio lands directly comparable to E-051's 0.0330. **#503's DECISIVE NUMBER LANDED at 11:30Z — day-matched corridor AUC 0.944**
(vs monthly 0.939, pentad-K24 -0.499) — **and its own skill-vs-lead profile
says REPLAY.** 0.971 at 5 days, 0.949 at 30, 0.942 at 90, 0.946 at 365: FLAT,
where forward physics decays; and `msss_pers` 0.966 at a one-year lead on a
z-scored anomaly field has no physical story. Mechanism: a 365-day roll from a
2009 holdout start walks into 2010, which is training data. Report it as a
corridor AUC UNCERTIFIED on two counts (no pentad gate reference; battery
incomplete) whose profile predicts it will not survive; and +0.005 over
monthly is a CONSISTENCY at a tier sd of 0.0020, never a beat. **The battery
is now the most informative measurement on the board** — if it confirms
replay, the monthly 0.939 champion is under the same question. **But #503
cannot deliver it**: `job_timeout` 2400 min kills the job at ~15:52Z 08-29, at
~48% of the battery, and the long roll writes only on completion. Riding buys
~$8.4 and no artefact. Cut-and-re-dispatch-sized-to-finish recommended to
Chris 13:35Z (long/future shortened 20 y -> ~5 y = 730 steps ~17.7 h, a flag
not a build). #500 (80k) / #502 (68k) FGN pair healthy.
*Every ML session updates this stamp and the sections it touches in
the same breath as harvesting or dispatching — the standing instruction is
`ml/CLAUDE.md` §0g. If the stamp is more than a day old, distrust the
"in flight" section and check the
[status page](https://blauewelt.github.io/earth/status.html).*

This page is the curated map: what question each experiment asked, what it
answered, and where the programme's next GPU-hour should go. The
authoritative record behind every line is
[the experiment log](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md);
the live fleet is the
[status page](https://blauewelt.github.io/earth/status.html); the narrative
synthesis is
[the paper](https://github.com/blauewelt/earth/blob/main/ml/paper/paper.pdf).

## 0 · The programme in one paragraph

Encode everything the observing system knows about each ocean pixel into an
embedding (stage 1), predict forward in embedding space (stage 2), read out
anything — AMOC at 26.5°N is the headline read-out because RAPID is the best
truth series, not because transport is the target. A representation is
scored by what stage 2 predicts from it; the embedding's job is
attendability, never information beyond the pixels.

## 1 · In flight right now

| what | TL;DR question | must beat / registered reading | where · ETA |
|---|---|---|---|
| **#504** E-056a-R token substrate K=24 (+ E-056a-CLEAN twin registered) | is the E-050 warm-FSQ 16-bit-per-pixel-bin alphabet a competitive forecasting substrate? Overnight both arms died: #494 CANCELLED at step 2800/20000 by an actor outside this session; #495 VOID — CUDA OOM on its first forward (K=144 batch 256 does not fit a 24 GB 4090; its control #478 ran that batch on a bigger card, so E-056b needs an 80 GB box, NOT a halved batch). Codec now durable on the release; token-Z cache durable ⇒ embed is free | ≲0.44 ⇒ token road opens at ~5% state size; ≳0.50 ⇒ quantization lost the forecastable signal. Denominator is the TOKEN-scale 0.63451. znoise-dose confound pre-registered (0.879 vs 0.151 rel_pers) — the CLEAN twin at znoise 0.12 settles it in the same wave | **TRAINING CURVE DONE 09:42Z: final ratio 0.53873** (0.34183/0.63451) after 0.600@2800 / 0.574@6000 / 0.548@16000 — flat from ~16k, WORSE than the continuous control 0.5056 and far from the 0.4394 lattice bar. Not a verdict: ~5.8x the intended relative dose. Probe ladder tail still running |
| **#505 re-dispatch** E-056a-CLEAN, K=24, znoise **0.12** | is #504's 0.555 the SUBSTRATE, or the 5.8x noise handicap? Identical to #504 in every other field | the dose-matched level against lattice 0.4394 / continuous 0.5056, denominator 0.63451. If CLEAN materially beats 0.7 the dose was the handicap; if they agree the substrate verdict stands at either dose | **RE-DISPATCHED as #507 at 10:19Z, QUEUED on gpu-box-32966687 behind #504** — deliberately the SAME box as its 0.7-dose twin, so the pair differs in the dose and nothing else (three other hosts were tried and failed: one full disk, two `resources_unavailable`, one that never registered its runner in 18 min). **RUNNING since 11:24Z, healthy at 0.306 s/step, gpu 99.99%** — ratio 0.590 at step 2,800 against #504's 0.600 at the same step, so the dose looks like a small effect so far; done ~13:10Z plus its ladder |
| **#506** E-056b token substrate at K=144 — **CANCELLED 12:30Z, HELD** | does the dense two-year slab hold up on tokens, and what does it cost? | vs #478's 0.0820 — unanswered | the rented H100 never used its GPU: ten `gpu_util` samples of 0% at `cpu_util` 95-97% across 3 h 55 m (the dead-frame check never fired, so they were real), and zero step records in 50 min where #478 at identical geometry writes its first at `wall_s 240`. Cancelled on a rule written 25 min ahead; ~$7.95. **Re-dispatch only if #507 rehabilitates the substrate** |
| **#500/#502** E-057.1 FGN pair, NOW PARALLEL (seed 0 on gpu-box-42005419 since 19:21Z; seed 1 re-dispatched by the stencil session as #502 on fresh gpu-box-46292015, fixed sha) | does a LEARNED perturbation + fair CRPS (eps^32 conditional LN, N=2, znoise OFF) replace the hand-dosed znoise and un-damp the roll? | ensemble-mean corridor AUC vs znoise pair 0.7235 (F1); stage2_val_member_var -> 0 = eps collapse (F2 — #500 reads 0.5–0.77 at 8k, healthy); ensemble-roll evaluator BUILT and ready | both ~27 h ⇒ pair lands ~24Z 08-28 (was ~2 days sequential); >24 h token expiry = HAND-HARVEST both; cross-box caveat in e-057(j) |
| **E-054b** ~400M capacity rung (1280×20, K 144, 200k fresh, TPU spot, grad-accum 4) | does capacity, not steps, buy the next factor at full pentad span? | vs E-051's 0.0330 at 200k/206.6M (the step-matched control — E-054a's 0.02981 is a 400k number); first-minutes verdict **PASSED on knobs** (`K 144 · lr 4e-4 · 1280x20 · stencil 145 · znoise 0.7 · grad_clip 128 · grad_accum 4 (micro 64)`, Z pulled and VERIFIED 16.24 GiB); **and the first training step at 08:37Z showed NO OOM** — params_M 399.948, grad_accum 4, micro_batch 64, and val_persistence 21.44621 bit-identical to E-054a's denominator, so the ratio is directly comparable to E-051's 0.0330 | first launch OOMed 00:15Z (registered risk fired) ⇒ grad-accum built + certified exact (2.4e-07) and pushed; a 07:08Z relaunch ran the WRONG config for ~1 h and was killed; **relaunched 08:10:33Z us-west1-c spot, correct** — ≈32 h ⇒ ~16Z 08-29 |
| **#503** E-051 roll (398k K=144 head, day-matched, replay battery, FIRST roll with E-055's unpooled keys) | does the best pentad one-step ever (0.0298) survive a 12-month roll, where the small pentad head collapsed to −0.499? | vs monthly _trainlon 0.939/0.939 and pentad-K24 −0.499; the battery (tracking-vs-lead profile; flat = calendar replay) is MANDATORY before the number may be called forecast | measured 06:50Z: 87.1 s/step. **Decisive corridor AUC at step 657/3363 ≈ 16:50Z 08-28**, inside the 23:52Z token; the 2,922-step battery needs ~71 h beyond it — ride-vs-cut is a decision for Chris |
| **#485** E-050 warm-start FSQ | does a trained encoder survive quantization where every cold start collapsed? | decoder-ceiling audit (Falsifier B): fast channels inside the 9–19% FVU band on Argo-free bins | **finals ARCHIVED** (run-485.jsonl + probes-485.json on ml-metrics) — read-out pending its own session's harvest |
| **E-052.1** det field head (diffusion session's run) | can one model predict the whole field jointly? | died at its first ckpt save (torch-less train venv); **RELAUNCHED 05:44Z by its session, resumes from step 1000**, holds the on-demand v5litepod-4 | its session harvests; finish ≈04Z 08-28 if pace holds |

## 2 · Most promising next steps, ranked

1. **E-051's roll decision** (its own session). One-step landed at
   **0.0330** (0.70869/21.44621) vs its 20k twin's 0.0820 — full budget pays
   2.5× at K=144, the best pentad one-step ever. With fusion eliminated, the
   span hypothesis carries the pentad-roll question alone; the roll (replay
   battery first) is the remaining read-out.
2. **E-057.1 · the FGN pair** (Chris: *"let's prioritize an experiment
   with: 1. Noise-conditioned stage-2 head trained with fair CRPS"* —
   2026-08-27). TL;DR: does a LEARNED perturbation + a proper score replace
   the hand-dosed znoise and un-damp the roll, at one forward pass per
   member? E-057.0 is BUILT and CPU-verified (`--fgn-eps` in temporal.py;
   zero-init identity bitwise; loss pinned to probscore; shared-coin toy:
   coherence 0.99 shared-ε vs 0.15 independent). E-057.1 = two seeds at
   monthly xl144, znoise OFF, vs clean 0.6781 and znoise 0.7235 (both
   two-seed controls); falsifiers F1–F3 pre-registered; the ensemble roll
   (M members, ε per step) is BUILT + CPU-verified — the corridor read
   runs when the pair lands.
   [Plan](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E057_fgn_head.md) ·
   [FGN addendum](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E052_FGN_addendum.md) ·
   [log entry](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-057).
3. **E-050 decoder-ceiling audit** once #485's finals land — decides whether
   one warm 16-bit token carries a pixel-bin; a pass opens token-input /
   token-output heads on the best forecasting substrate found so far
   (lattice z, E-046).
4. **E-053 space-time stencil** (Chris's direction, RUNNING): E-053.0
   measured — the advective cone is refuted (ridge lag 0 everywhere; the
   agnostic ball/ramp wins) and the analog is visible in the substrate.
   **A1 (#486): 0.1858. A2 (#487): 0.1561. Both between the registered
   thresholds — the span effect is genuinely DISTRIBUTED.** A long span
   helps under any sampling (0.5056 → 0.156–0.186; pins alone 55%, log
   ramp 65% of the effect in log terms), but the dense slab keeps a 1.9×
   edge (0.0820) that no 16–24-frame skeleton found. Still open: A3
   (10-y span, running next) and A4 (uniform spacing, registered 05:50Z —
   isolates spacing from sparse K). Replay battery mandatory before any
   offset head is rolled; rollout_spatial refuses them until offset-aware
   assembly exists.
   [Plan](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E053_spacetime_stencil.md) ·
   [slides](https://blauewelt.github.io/earth/ml/figures/spacetime_stencil.html).
5. **E-052.1 deterministic field head** (diffusion session) — the
   architecture-level alternative; param-matched 200.4M. **Currently
   stalled at step 960 (§1) — its session's first task is diagnosis and
   resume before any new spend.**
6. **If E-051's roll is negative:** the hierarchical fallback — roll at
   monthly stride *through* the pentad stack (A2a's 0.0721 head is that
   object) — plus a pentad-calibrated znoise dose sized from the pentad
   one-step error (A4's lesson), before new architecture spend.
7. **Housekeeping by expiry:** publish-or-write-off pixelmae-472/-473/-477
   (30-day clock, 2026-09-24); slice-the-published-Z (unblocks E-045.3 and
   is E-053.1's own code path); the #472/#473-vs-TPU cross-framework Tier-1
   certificates; E-046's owed seed-0 refinish.

## 3 · Completed experiments, one line each

### Monthly foundations (E-001 … E-021) — mostly closed

| ID | question | verdict |
|---|---|---|
| E-001/002/003 | does pretraining / longer training / capacity help the probe? | not established / null / null |
| E-004 (a,b,v5) | joint stage-1+2 training | all RETRACTED — loss rewarded z-shrinkage; instrument defects |
| E-005→E-010, E-016, E-020 | is unroll the axis? | axis CLOSED with replicates; the +0.28 was seed noise (the founding cautionary tale) |
| E-007/E-008 | is stage 2 compute-bottlenecked? | no — forecast improves, probe flat |
| E-019 | codec round-trip loss | audit instrument, still in use (Tier-1 descends from it) |
| E-021 | 20-year ensemble fan | hindcast skill was memorisation (r +0.42 held-out, ~10× under-dispersed) |

### Geometry, scale, and the monthly champion line (E-022 … E-037, E-043)

| ID | question | verdict |
|---|---|---|
| E-022–E-027 | spatial coupling; ring/spiral/sunflower geometry; capacity | capacity breaks the 34M inversion; +0.042 per tier to 205M, no saturation |
| E-028–E-032 | the width ladder at xl | closes at 144 points (0.6781); 233 buys nothing (0.6739 pair) |
| E-035/E-036/E-037 | width reopened; does znoise survive scale; noise×width | znoise +0.045/+0.050 at 205M — the largest single effect; noise makes width irrelevant (0.7235/0.7240) |
| E-038 series | read-out discipline at pentad | pooling destroys the transport signal; head-vs-matched-raw is the protocol; codec ≈ raw at the current-state probe (parity, not a verdict on the codec's job) |
| E-043 (5 arms) | retire the 45–25°W longitude holdout | trained-vs-held gap 0.65 → 0.0065; corridor 0.939 replicated n=2 — but flat lead-time profile and no transport movement: not readable as forecast skill |
| E-043b-PHASE | calendar or context? | CALENDAR REPLAY — long rolls recite the record; every long-hindcast r withdrawn |

### The cadence programme (E-044 … E-045, E-047, E-051)

| ID | question | verdict |
|---|---|---|
| E-044/E-044b | first pentad stage-2 | needs grad-clip 128 (#423 diverged); one-step 0.5056/0.5045 (pair) |
| E-044b-roll | does the pentad head roll? | **−0.499, below climatology** (n=1) — the wound the frontier addresses |
| E-045 factorial | which component breaks at pentad? | **span, not step**: span-fixed 0.0721 · 0.0804 · 0.0820 flat vs K-fixed 0.07→0.51 cliff; mechanism registered = seasonal analog. Side: Argo targets stabilize (A3), monthly noise dose fatal (A4 0.81), season staircase null (A6) |
| E-045.3 | the K=48 rung | BLOCKED — config-tied CPU-fall ×2; unblock = slice-the-published-Z |
| E-053.0 | is there an advective cone in the z field? | NO at the argmax instrument (ridge lag 0 to 2,500 km) — ball form wins; analog bump measured (0.143 @1 y vs 0.100 @180 d deseas.) |
| E-053.1-A1 | is span's value the seasonal analog? | **0.1858 — between the registered thresholds** (≲0.12 / ≳0.3): pins buy 2.7× over uniform K=24 at the same frame budget, ~55% of the span effect (n=1) |
| E-053.1-A2 | does log spacing match the dense slab at 1/6 the frames? | **0.1561 — no**: 65% of the span effect, still 1.9× short of K=144's 0.0820. With A1: the effect is distributed across the slab; sparse skeletons don't find it. A4 isolates spacing (n=1) |
| E-053.1-A3 | does context beyond 2 years carry information — the first 10-y span ever? | **0.1400 — the decade is NOT empty**: +10.3% over A2 at the same construction (frames confounded with span; A4 calibrates). Best sparse arm; does not beat the dense 2-y slab's 0.0820 (n=1) |
| E-053.1-A4 | is it the SPACING or the sparse K that loses to the dense slab? | **0.14116 — spacing is irrelevant** (#489, 17Z 08-27): uniform ≈ log ≈ decade-pins (0.141/0.156/0.140) at K 24–32, all 1.7–1.9× short of dense K=144's 0.0820. **Wave verdict: frames, not placement, are the binding resource** — the span effect is distributed, sparse skeletons recover 55–65%, and the cheap road to span is more frames cheaper (E-056's case). E-053.2 point-cloud build demoted to the field-head unification (n=1, all arms) |
| E-047 | fusion vs selection (month-block codec) | Tier-1: Argo-anchor collapse cured at a 9–19% everywhere-cost; **stage 2: fusion LOSES, 0.2127 vs 0.0721** (#483, 08-26) — block-decode roll not dispatched |
| E-051 | span at full budget | **one-step 0.0330** vs the 20k twin's 0.0820 — budget pays 2.5× at K=144, best pentad one-step ever; roll decision pending (its own session) |
| E-054a | does doubling the training budget (200k→400k, re-armed LR) keep paying at K=144? | **0.02981 @400k** (from 0.0330) — still falling, decelerating: steps pay with diminishing returns, capacity is the indicated axis (E-054b launched). Artefact note: durable final is 398k (no post-loop save) |

### The quantization road (E-046, E-048, E-049, E-050)

| ID | question | verdict |
|---|---|---|
| E-046 | lattice z vs continuous z as forecast substrate | **0.4394 < A9's 0.4916 < 0.5056** — training through the lattice wins; priced: gradient spikes killed one of two seeds |
| E-048 | window blocks + fitted FSQ ladders | fitted ladder closes collapse, not drift; unbounded FSQ wears 8 levels as a sign code |
| E-049a/b | one 16-bit token per pixel-bin | (a) continuous d_z-6 control healthy to 200k; (b) cold-start FSQ = constant-encoder collapse ×2 — the lattice-at-cold-start is convicted |
| E-050 | warm-start quantization | #485 finals archived (run-485.jsonl on ml-metrics); decoder-ceiling audit read-out pending its own session's harvest — see §1 |

### Data & infrastructure experiments

| ID | question | verdict |
|---|---|---|
| E-033/E-034, E-039/E-040 | data programme: tensors, daily SST plumbing | pentad/daily tensors built; r2 adds SST as channel 40 |
| E-042 | what is SST worth alone? | **unanswered** — the matched pair was cancelled in triage and never re-run; every r2 codec carries the channel as an untested assumption |
| JAX port / TPU tier | a second implementation | gated at 1e-7…1e-5; stage-1 AND stage-2 trainers ported; TPU ≈4.5× H100 per sample; tier never pooled with torch |
| **E-058** rung 1 | can a roll be read for SST, not just for the 40-channel pool? | **BUILT + CERTIFIED, no GPU.** `chan_skill` was per-HORIZON and pooled over all 40 channels, so `sst` was 1/40 of an undecomposable number. `rollout_spatial.py` now emits `per_channel` beside it from the SAME `_skill_rows`: pooled bytes identical (18,289), consistency 1.11e-16, known-answer +1.000 / −1.025 where the pool says +0.310. Rungs 2–3 owed: SST in `head_targets`, then Florida Current as a third, instrument-independent target |

## 4 · Standing cautions (the short list a new reader needs)

Unpooled head numbers are the verdict, pooled are labelled legacy. n=1 is a
direction, never a level — one seed is licensed only at the monthly xl
corridor tier (sd 0.0020); every pentad/daily number except the E-044b pair
is unreplicated. Long rolls are replay; any head with analog access must
pass the E-043b-PHASE battery before its roll is quoted. Jobs >24 h lose
their archive step to token expiry — harvest by hand. Ops expectations live
in the project doc `claude/expectations.md` (fleet artefacts, boxes, holes).

# The standing overview — every experiment, one line each, and what's next

**Last updated: 2026-08-27 ~17:55 UTC** (stencil session: **E-053 wave
RESOLVED — A4 0.14116; synthesis: frames, not placement, are the binding
resource**; E-056 first pair #490/#491 died on the fsq-warmstart resume
guard, **re-dispatched 17:44Z as #494/#495** on the base fsq65k recipe;
E-054a mid-check 320k @ ratio ≈0.0312, finish ~22:30Z; #492 (E-057.1 seed
0) FAILED and its box reads offline — its session diagnoses; #493 running
on gpu-box-32966687. Carried from FGN session's 18:05-stamped edit:
E-057.0 cross-verified against DeepMind's open-source FGN — loss and
conditional-norm arithmetic match exactly, deviations in
EXPERIMENTS.md#e-057(g).) *Every ML session updates this stamp and the sections it touches in
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
| **#494/#495** E-056a/b token substrate, RE-DISPATCH (queued behind #493 on gpu-box-32966687) | is the E-050 warm-FSQ 16-bit-per-pixel-bin alphabet a competitive forecasting substrate — and what does K=144 cost on it? First try #490/#491 died on the fsq-warmstart resume guard (recipe carried the flag; guard correct); re-dispatched 17:44Z on the base fsq65k recipe. Free finding: the 260k FSQ final is confirmed present box-locally (guard read its args) | a: ≲0.44 ⇒ token road opens at ~5% state size; ≳0.50 ⇒ quantization lost the forecastable signal. b: ≈0.08 + faster steps ⇒ next full-budget head trains on tokens | behind #493's ~30 h — results likely 08-28; gates the next $100-class pentad spend |
| **#492/#493** E-057.1 FGN pair (FGN session's runs) | does a LEARNED perturbation + fair CRPS (eps^32 conditional LN, N=2, znoise OFF) replace the hand-dosed znoise and un-damp the roll? | ensemble-mean corridor AUC vs znoise pair 0.7235 (F1); stage2_val_member_var -> 0 = eps collapse (F2, live branch) | **#492 FAILED ~17:26Z on gpu-box-40623952 — that Vast host now reads OFFLINE** (its session diagnoses); #493 (seed 1) running on gpu-box-32966687 since 17:38Z, ~30 h |
| **E-054a** continue E-051 → 400k (this session, TPU **spot**, node `e051-k144-full`) | does the unsaturated budget curve keep paying past 200k (LR re-armed: 4e-4, halflife 100k)? | vs 0.0330: ≈0.026 ⇒ budget still paying, queue the ~400M capacity rung (E-054b); ≈0.033 flat ⇒ capacity is the axis | **mid-check 17:39Z: step 320k, val ratio ≈0.0312 and falling** — trending between the registered poles (paying, diminishing); pace ⇒ 400k finishes **~22:30Z tonight**; frozen-200k copy safe in the bucket |
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
   two-seed controls); falsifiers F1–F3 pre-registered, the ensemble-roll
   diff (M members, ε per step) is the remaining build item before its
   corridor read.
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

## 4 · Standing cautions (the short list a new reader needs)

Unpooled head numbers are the verdict, pooled are labelled legacy. n=1 is a
direction, never a level — one seed is licensed only at the monthly xl
corridor tier (sd 0.0020); every pentad/daily number except the E-044b pair
is unreplicated. Long rolls are replay; any head with analog access must
pass the E-043b-PHASE battery before its roll is quoted. Jobs >24 h lose
their archive step to token expiry — harvest by hand. Ops expectations live
in the project doc `claude/expectations.md` (fleet artefacts, boxes, holes).

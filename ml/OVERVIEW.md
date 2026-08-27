# The standing overview — every experiment, one line each, and what's next

**Last updated: 2026-08-27 ~05:55 UTC** (by the E-053 session, at Chris's
check-in: A1 0.1858 and A2 0.1561 harvested; A4 registered + dispatched
(#489); E-051's final 0.0330 entered; E-052.1 found stalled at step 960 —
flagged, its own session owns it). *Every ML session updates this stamp and the sections it touches in
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
| **#488** E-053.1 A3 decade-32 (next on gpu-box-32966687, behind #487's probe tail) | does span keep paying beyond 2 y — the first 10-y context ever? | <0.0820 ⇒ decadal territory; ≈0.082 ⇒ 2 y saturates. Anchor pool shrinks ~23% by construction | result ~10Z 08-27 |
| **#489** E-053.1 A4 uniform-24 (queued behind #488; registered 05:50Z after A1+A2) | is it the SPACING or the sparse K that loses to the dense slab? | ≈0.15 (A2-like) ⇒ spacing irrelevant, slab density load-bearing; ≲0.10 ⇒ uniform beats log — E-045's flatness partly a spacing artefact | result ~14–15Z 08-27 |
| **#485** E-050 warm-start FSQ | does a trained encoder survive quantization where every cold start collapsed? | decoder-ceiling audit (Falsifier B): fast channels inside the 9–19% FVU band on Argo-free bins | **finals ARCHIVED** (run-485.jsonl + probes-485.json on ml-metrics) — read-out pending its own session's harvest |
| **E-052.1** det field head (diffusion session's run) | can one model predict the whole field jointly? | **⚠ STALLED — metrics stopped at step 960/24,000 (00:22Z 08-27, loss 14.6); mirror re-run 05:10Z confirms no progress; finals show only ckpt_latest.npz.** Needs its session's diagnosis + resume/redispatch — do not assume the ≈20:45Z finish | flagged 05:15Z; own session owns it |

## 2 · Most promising next steps, ranked

1. **E-051's roll decision** (its own session). One-step landed at
   **0.0330** (0.70869/21.44621) vs its 20k twin's 0.0820 — full budget pays
   2.5× at K=144, the best pentad one-step ever. With fusion eliminated, the
   span hypothesis carries the pentad-roll question alone; the roll (replay
   battery first) is the remaining read-out.
2. **E-050 decoder-ceiling audit** once #485's finals land — decides whether
   one warm 16-bit token carries a pixel-bin; a pass opens token-input /
   token-output heads on the best forecasting substrate found so far
   (lattice z, E-046).
3. **E-053 space-time stencil** (Chris's direction, RUNNING): E-053.0
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
4. **E-052.1 deterministic field head** (diffusion session) — the
   architecture-level alternative; param-matched 200.4M. **Currently
   stalled at step 960 (§1) — its session's first task is diagnosis and
   resume before any new spend.**
5. **If E-051's roll is negative:** the hierarchical fallback — roll at
   monthly stride *through* the pentad stack (A2a's 0.0721 head is that
   object) — plus a pentad-calibrated znoise dose sized from the pentad
   one-step error (A4's lesson), before new architecture spend.
6. **Housekeeping by expiry:** publish-or-write-off pixelmae-472/-473/-477
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

# Experiment log

Every experiment, what it tested, what it returned, and what we concluded —
newest first. Chris asked for this on 2026-08-09: *"please keep a log of all
experiments and their results."*

**Rules for this file.**

1. One entry per experiment, written **when it is dispatched** (hypothesis
   first, so the log cannot be rewritten to fit the answer) and completed when
   it lands.
2. Every entry names the **code it ran**: run number → `head_sha`. The status
   page links it. A result whose code cannot be recovered is an anecdote.
3. State the **control** explicitly. A number without the baseline it must beat
   is not a result.
4. Record **negative and null results with the same care as positive ones**.
   Most of the entries below are null, and they are the reason the programme
   knows where its bottleneck is.
5. Numbers here are the **year-blocked k-fold** figures — but *which* k-fold
   depends on what you varied, and conflating the two cost a four-arm
   experiment on 2026-08-10. `probe_kfold.py` scores the **CODEC**: it pools
   the frozen embeddings and never sees the temporal head, so every run
   freezing the same codec returns the same number (#116 and #125, a 60k and
   a 200k head, both read 0.631 [0.513, 0.732]). Stage-2 questions —
   schedule, budget, unroll — are answered by `rapid_probe_kfold` in
   `temporal.json`, the same protocol over the head's own features.
   In-training light-probe values, and `rapid_probe`'s 36-month single split,
   are noted as such wherever quoted. See `docs/ML_BASICS.md` §5b.

Baselines, for reference. Wind-stress-only ridge in our own protocol:
**0.531** on the 1° tensors, **0.568** on the quarter-degree tensor. RAPID
monthly σ = 2.79 Sv, n = 240, ~68 effective DOF (~9 after an 18-month
low-pass).

---

<a id="e-015"></a>
## E-015 · Stage-2 WIDTH: the parameter-bottleneck arm — PREPARED 2026-08-11, dispatch after E-013

**Why, from Chris.** *"In your next experiments, please consider trying out a
larger stage 2 model (u=1) as well."* This is the arm E-008 explicitly left
open: 33× compute moved the forecast objective steadily and the AMOC probe
not at all, which closed the compute bottleneck and said nothing about
CAPACITY. Nobody has ever trained stage 2 at more than d_model 192 / 4
layers.

**Design.** d_model 384 / 6 layers (~5.9× the parameters), from scratch,
U=1, 60,000 steps, `sched:expdecay --lr-cooldown-frac 0` (horizon-free, hot
endpoint — extendable without a warm restart), same codec, same tensor
`adcbe700`, seeds {0, 1, 2}. Everything else pinned to E-012's U=1 arms,
which are the direct comparison set.

**Pre-registered questions.** (1) Does width move `rapid_probe_kfold` above
the U=1 seed band at the same budget (E-012: 0.363/0.446/0.493, sd 0.066)?
(2) Does it move the rollout/AMOC-at-horizon metrics (seed-stable ±0.003,
so 3-vs-3 resolves small effects) once E-013's protocol reads it? (3) Does
the z-ratio drop below the ~0.39 plateau that has now reproduced across
schedule and tensor?

**Falsifier.** All three flat → capacity was not the binding constraint at
this data size either, and the bottleneck moves definitively to the
OBJECTIVE (E-006) and the embeddings themselves.

**Dispatched 16:00Z as #180–#182** (seeds 0/1/2) — cancelled unstarted in
the disk incident (see E-014's re-dispatch note), re-queued as #188–#190,
then re-pinned to the new box per Chris as **#193–#195** (final).

**Result.** *pending.*

---

<a id="e-014"></a>
## E-014 · DIRECT multi-horizon heads: predict t+h without iterating — PREPARED 2026-08-11, dispatch after E-013

**Why.** E-011 closed the unroll axis and measured WHERE the rollout loses:
amp_ratio < 1 at every horizon — the iterated path feeds its own smoothing
back in, and the AMOC probe reads exactly the amplitude that is lost. The
2026-08-11 code audit (E-011 scope note) identified the un-detached BPTT
gradient as the leading mechanism. The direct alternative never iterates:
one linear head per horizon predicts z_{t+h} from the hidden state at t in
a single forward. Nothing compounds because nothing is fed back.

**Design.** `--direct 3,6,12` on the E-012 U=1 trunk (d_model 192, 4
layers, 60k, expdecay no-taper, same codec, same tensor), seeds {0, 1, 2}.
Loss adds mean-over-horizons last-position MSE; `--direct` empty is
bit-identical to the old objective (tests/test_direct_heads.py asserts it).
The instrument is `rollout.py`'s PAIRED comparison, built and toy-tested
before dispatch: direct and iterated scored in the same loop iterations on
the same starts, target months and observed cells
(`delta_msss_clim_vs_iterated`), plus AMOC through the truefit probe and a
directfit probe fit on the direct heads' own train-year predictions.

**Hypothesis.** Direct wins at its trained horizons — most at h=12, least
at h=3 — with amp_ratio nearer 1, and the AMOC read at horizon improves
accordingly.

**Falsifier.** Paired deltas ≤ 0 with amp_ratio no better → the horizon
loss is not an artifact of iteration; it is in the embeddings or the
predictability itself, and no readout change will buy it back.

**Dispatched 16:00Z as #177–#179** (seeds 0/1/2), plans published,
queued behind E-013 (#176). **Re-dispatched 17:45Z**: the box disk hit
50/50 GB mid-#177 (hygiene freed the published Z, the re-pull refused
for headroom, #177 fell back to an unpersistable in-RAM rebuild and went
metrics-blind — every write to disk fails silently while the compute is
fine). #178/#179 were cancelled unstarted; a SECOND box (Vast 47483091,
100 GB disk, per Chris) was rented, the workflow now seeds the PINNED
tensor `adcbe700…` from data-cache-v1 with sha verification (so
cross-box arms are identical by construction), and seeds 0/1/2 re-queued
as **#185–#187**. #177 itself runs to completion — if its API-side
uploads survive the full disk it is a valid (duplicate) seed-0 arm.
**Final run numbers (17:55Z, Chris: "all relevant experiments on the new
box; box 1 is for investigation")**: #186/#187 cancelled and re-dispatched
PINNED to `gpu-box-40623952` as **#191/#192** (seeds 1/2); #185 (seed 0)
was already running there. The old box runs only #177 and the #184
triage until its disk is understood.

**Result.** *pending.*

---

<a id="e-013"></a>
## E-013 · The rollout protocol over the CONVERGED heads, with the upgraded instrument — DISPATCHED 2026-08-11 16:00Z as #176

**What runs.** `rolleval:e012_u1_s0,e012_u1_s1,e012_u1_s2,e012_u4_s0` —
all four 60k heads (extracted from their run artifacts by
`scripts/publish_heads.mjs`, which verifies each checkpoint's own args
against the claim before naming the asset), through `rollout.py` as
upgraded today:

- **rolledfit probes**: AMOC ridge fit per horizon band on ROLLED
  train-year section states, read on holdout rolled states — removing the
  truefit probe's train/apply distribution shift (rolled states are
  smoother; E-011's amp_ratio < 1);
- **3-seed ensembles** of per-point probe predictions for the U=1 group,
  joined on (horizon, start) keys.

**Pre-registered questions.** (1) Does convergence change E-011's
every-horizon falsification of unroll? (single U=4 arm; the rollout
metrics' ±0.003 seed stability is what makes one arm readable.) (2) How
much of the nowcast-to-horizon drop (0.63 → 0.31 at h1–3) does the
rolledfit probe recover? (3) Does the 3-seed ensemble beat the best single
seed?

**Deviation, recorded**: an eval run trains nothing and has no LR schedule,
so no plan file — the E-011 precedent.

### RESULT (#176, 16:20Z) — field falsification holds; the deep-horizon AMOC read flips; rolledfit BACKFIRES

**Q1 — convergence does not rescue unroll on field skill, but the picture
is no longer uniform.** Horizon AUC: U=1 0.266/0.291/0.282 vs U=4
**0.236** — still last, though the gap has narrowed from 6k (0.165–0.168
vs 0.229–0.240). E-011's every-horizon field falsification stands at
convergence. The AMOC bands, however, no longer agree with it at depth
(truefit probe):

| head | h1–3 | h4–6 | h7–12 |
|---|---|---|---|
| u1_s0 | 0.414 | 0.182 | 0.177 |
| u1_s1 | 0.365 | 0.082 | 0.183 |
| u1_s2 | 0.437 | 0.201 | 0.300 |
| **u4_s0** | 0.394 | **0.278** | **0.335** |

The single U=4 arm reads best-in-set on both deep bands. One seed, a
probe whose seed noise is 0.066–0.123, and E-010's (unestablished)
observation that U=4 compresses probe variance — so this is a DIRECTION,
not a result. But it is the first time anything has moved the deep-horizon
transport read in unroll's favour, and it happened only at convergence.
Also: convergence helps everyone — U=1 seed-means improved from E-011's
0.31/0.11/0.21 to ~0.40/0.16/0.22.

**Q2 — the rolledfit probe does not recover the drop; it makes it worse.**
h1–3: 0.234–0.368 vs truefit's 0.365–0.437; h4–6 goes NEGATIVE for two of
three U=1 seeds (−0.286, −0.371). Post-hoc reading: the band fit pools
staggered starts whose contexts overlap heavily, so its 561 "points" are
far fewer effective samples than the lambda-selection tail assumes — an
under-regularised ridge on model-generated features that do not
generalise across the year-blocked split. The distribution-shift argument
was plausible and is now measured wrong: **the truefit probe remains the
instrument.** (Follow-up if ever needed: year-blocked lambda selection
inside the band fit. E-014's directfit probe shares this machinery — read
it with the same caution.)

**Q3 — ensembles track the seed mean, no magic.** u1 truefit ensemble
0.414/0.156/0.227 — above the seed mean, below the best seed everywhere.

**E-012 is hereby CLOSED**: seed noise at 60k = 0.066 (halved from 6k,
still probe-dominated); U effect on the probe unmeasurable as designed
(two arms cancelled, deviation recorded), directionally interesting at
depth via E-013; one-step cost persists in full (+27%). The unroll axis
stays closed for training the FORECAST — the open question it leaves is
narrower and new: does unroll's variance compression happen to help the
TRANSPORT probe at depth? Answerable by two more U=4 seeds (~70 min GPU)
if the direct heads (E-014) don't render the question moot first.

---

<a id="e-012"></a>
## E-012 · The unroll sweep at 60,000 steps — DISPATCHED 2026-08-11, queued behind E-011

**Why, from Chris.** E-010's arms ran 6,000 steps — where E-007 puts the
z-ratio at 0.494 against 0.391 at 60k, i.e. visibly undertrained — and four
of its six arms sat below the wind bar. *"Rerunning the U sweep with more
steps (60k) would make sense."* So: the same design at near-convergence.
U ∈ {1, 4} × seeds {0, 1, 2}, 60,000 steps, cosine to 1e-3, one box, one
tensor, scored on `rapid_probe_kfold`. Six runs, ~5 h of GPU.

**Pre-registered questions, in order of value.**

1. **Is the probe's seed noise training-dependent?** E-010 measured sd 0.123
   at 6k. If it stays ~0.12 at 60k, the instability is probe-intrinsic
   (240 months, ~9 effective DOF) and no amount of training buys resolution —
   which caps what ANY stage-2 sweep can ever show on this target. If it
   shrinks substantially, converged heads are comparable in a way 6k heads
   are not, and E-010's null gets a caveat.
2. **Does U matter at convergence?** E-010: +0.023 (t = 0.31) at 6k. Same
   test at 60k, three seeds a side.
3. **Does the one-step cost of unroll persist?** −29.7% at 6k; at 60k the
   U=1 z-ratio should land near E-007's 0.391, and the U=4 gap is measured
   against that.

**Falsifier for the unroll axis, final form:** U=4 within seed spread of U=1
on the probe AND still paying a large one-step cost at convergence — after
which E-013 (rollout eval of these heads, the E-011 protocol) either finds
the multi-step payoff or the axis is closed at every horizon *and* every
budget, and `--unroll` becomes a documented dead end rather than a default.

**REVISED before any arm started (Chris): expdecay, no terminal taper.**
*"Maybe it would make sense to do the sweep with an LR regime that doesn't go
to zero."* The six arms are **#164 and #171–#175**,
`sched:expdecay --lr-cooldown-frac 0`: cosine warmup then `2^(−s/40000)`,
ending at **3.66e-4 = 36.6% of peak**. (Originally dispatched as #164–#169;
after #164 completed, the five still-queued arms were cancelled unstarted at
10:13Z so #170 — the E-011 retry — could jump the FIFO queue and answer
first, then re-dispatched with identical inputs as #171–#175. Any rescue or
sweep-table read must use the new numbers; #165–#169 contain nothing.)

What this buys: the schedule is horizon-free AND the endpoint is live, so
every head is a true continuation candidate — snapshots carry optimiser and
schedule state, and the expdecay prefix is invariant to extension. Extending
any arm to 200k later is a genuine continuation, not an E-008-style warm
restart with a fresh schedule.

What it costs, stated now: the endpoint is hot (36.6% of peak, where cosine
would be near zero), so final z-MSE will read noisier than a cosine-60k
head's, and E-012's z-ratios are not directly comparable to E-007's cosine
points — which were on another tensor anyway. All comparisons that matter
(U=1 vs U=4, seed spread) are internal and share the schedule. The cosine
plans for the first dispatch (#151–#156, then #158–#163) were cancelled
before any arm started; nothing trained on them.

**Sequencing.** Behind #170 (E-011, which answered — see below) on
`gpu-box-35586926`; the other two runners are deregistered, so FIFO onto one
box — and one tensor — is guaranteed by construction. Tensor `adcbe700` is
durable on `data-cache-v1` as of #150.

**First arm landed (#164, U=1 seed 0, 11:0xZ).** Head k-fold **0.363**
[0.208, 0.543], 36-mo split 0.416, z-ratio **0.3908** — which matches
E-007's 60k *cosine* figure of 0.391 across both a schedule change and a
tensor change, a reassuring stability check on the forecast objective.
Persistence 3.139…, provenance `adcbe700` ✓.

**DEVIATION (14:40Z, Chris): #174 and #175 cancelled unstarted.** *"If the
U=4 models have no hope of succeeding, you can also abort them, up to
you."* They have no hope of answering anything the programme still asks:
E-011 falsified the unroll at every horizon, the code audit identified the
mechanism as structural to the objective (the un-detached BPTT gradient
rewards smoothing — more steps train the same wrong incentive), and E-014's
direct heads supersede the idea. #173 (U=4 seed 0) was already training and
runs to completion as the single convergence check — read primarily through
E-013's rollout metrics, which are seed-stable to ±0.003, so even one arm
answers "does convergence rescue unroll?" with useful confidence. The
probe-level U question (3-vs-3) is forfeited, knowingly; the ~100 minutes
reclaimed go to E-014/E-015.

**The U=1 trio (question 1 answered).** #164/#171/#172, seeds 0/1/2:

| seed | head k-fold | CI95 | 36-mo split | z-ratio |
|---|---|---|---|---|
| 0 | 0.363 | [0.208, 0.543] | 0.416 | 0.3908 |
| 1 | 0.446 | [0.363, 0.552] | 0.438 | 0.3894 |
| 2 | 0.493 | [0.365, 0.608] | 0.407 | 0.3877 |

Seed sd **0.066** (range 0.130) against 0.123 (range 0.245) at 6k: training
to convergence roughly HALVES the probe's seed noise but leaves it ~20× the
field-metric's ±0.003 — the instability is mostly probe-intrinsic, as
E-010 concluded, with a training-dependent component convergence buys back.
The z-ratio is seed-stable to 0.003 across the trio and sits on the ~0.39
plateau. All three on `adcbe700` ✓ (and seed-0's persistence baseline
reproduces #121's to sixteen digits — the (tensor, seed) fingerprint is
exact).

**#173 landed (U=4 seed 0, 15:5xZ) — question 3 answered: the one-step
cost of unroll persists at convergence.** z-ratio **0.4953** against the
U=1 trio's 0.3877–0.3908: +27% worse at 10× the training, matching the
−29.7% measured at 6k. The objective does not grow out of it, which is
what the code audit predicted (the smoothing incentive is structural).
Probe k-fold 0.519 [0.375, 0.636], split 0.585, `adcbe700` ✓ — a single
arm, quoted with the caveat that it sits within reach of the U=1 seed
band (sd 0.066) and that E-010 observed U=4 compressing probe variance
(F = 9.5, never established). The probe-level 3-vs-3 U comparison was
forfeited with #174/#175; the rollout answer comes from E-013 (#176).

**Result.** CLOSED by E-013 (#176) — see that entry: seed sd 0.066 at
60k, one-step cost +27% persists, field falsification of unroll holds,
deep-horizon AMOC direction flips toward U=4 (one arm, unestablished).

---

<a id="e-011"></a>
## E-011 · Unroll DURING evaluation: rollout skill and AMOC at horizon — DISPATCHED 2026-08-11

**The question, from Chris.** E-010's z-ratio was teacher-forced horizon-1 —
the regime an unroll objective de-emphasises by design. The unroll's stated
purpose is surviving its own errors, and no evaluation ever fed it its own
errors. This one does: all six E-010 heads, autoregressive rollout into the
three holdout years, 12 horizons.

**What is measured, per head.**

- Channel-space skill at each horizon h (decoded predictions, observed cells,
  anomaly space) against three baselines: persistence, damped persistence
  (AR1, the literature's fair cheap baseline), climatology. Plus centred ACC
  and an amplitude ratio, and "horizon AUC" — mean skill-vs-climatology over
  h=1..12.
- **AMOC at horizon**: a ridge fit on TRUE train-month section embeddings,
  applied to ROLLED section embeddings — "given data to t, predict transport
  at t+h". Reported in bands h1–3 / h4–6 / h7–12, because single-horizon n is
  tiny.

**Hypothesis, written before the result.** If exposure-bias training does
what it claims, the U=4 heads should lose less skill per horizon step than
the U=1 heads — worse at h=1 (E-010 measured exactly that, −29.7%), crossing
over somewhere in h∈[2..6], and better in the AMOC h4–6/h7–12 bands. The
replicate rule applies: the comparison is three seeds against three seeds,
not one arm against another.

**What would falsify it (and close the unroll axis at every horizon):** U=4
at or below U=1 at every h, with the seed spreads overlapping. Given E-010,
this is the likely outcome — a 29.7% deficit at h=1 is a deep hole for
compounding stability to climb out of.

**Instrument caveats, stated at dispatch.** The AMOC bands are a single
train/apply split (not the k-fold), with the probe fit once on true
embeddings — treat band differences as direction, not measurement. E-010
measured the probe's seed noise at 0.245; the bands will be at least that
noisy.

**Mechanics.** `window: rolleval:e010_u1_s0,…` — the six heads were pulled
from their run artifacts, **verified against their claimed (unroll, seed, K,
steps)**, and published to `model-checkpoints-v1`. One embedding pass serves
all six. The toy test of this path caught that `rollout.py` has been unable
to load any K=24 head since it was written (`k_max=K` vs training's
`max(K, 36)`) — tonight would have been its first run and its first failure.

**Deviation from the graph-before-queue rule, recorded.** This run trains
nothing — there is no LR schedule to draw, so no plan file is published. The
rule certifies training schedules; an eval run has none.

### RESULT (#170, 2026-08-11 11:55) — **falsified at every horizon. Unroll training makes rollout WORSE, including the thing it exists for.**

Two dead dispatches first, recorded per the cost rule: #149 (the static pass
fed a patch=1 shape to a patch=3 codec) and #157 (k_max guessed from the
wrong convention — there are two, and the fix is to read the checkpoint's own
table). ~20 min of GPU across both; the run that worked cost ~10.

**MSSS vs damped persistence** (the fair baseline; positive = beats it):

| head | h1 | h2 | h3 | h4 | h6 | h9 | h12 | AUC |
|---|---|---|---|---|---|---|---|---|
| U=1 (3 seeds) | .171–.173 | .190–.194 | .200–.210 | .202–.212 | .205–.218 | .173–.185 | .094–.117 | .229–.240 |
| U=4 (3 seeds) | .031–.038 | .083–.092 | .115–.122 | .135–.138 | .152–.156 | .121–.123 | .018–.020 | .165–.168 |

**U=1 beats U=4 at every horizon, in every seed, on MSSS and on ACC** (h1:
0.652 vs 0.577; h12: ~0.40 vs ~0.32). The gap narrows mid-horizon and never
closes, never inverts. There is no crossover for exposure-bias training to
claim. The hypothesis said U=4 should win somewhere past h≈2–6; it loses
everywhere, including at the deep horizons that are its entire purpose.

**AMOC from rolled states**, noisier as pre-registered, agrees in direction:
U=1 seed-means beat U=4 in all three bands (h1–3: 0.31 vs 0.20; h4–6: 0.11
vs 0.09; h7–12: 0.21 vs 0.02).

**The unroll axis is now closed in full.** E-010: no probe effect at 6k
(t = 0.31) and −29.7% at horizon 1. E-011: worse at horizons 1 through 12,
worse on AMOC-at-horizon. The mechanism the flag was built on — train on
your own error distribution and compound more gracefully — does not
materialise: the U=4 model's degraded one-step predictor is what compounds,
and it never catches the U=1 model whose "unanticipated" errors turn out to
compound just fine. `--unroll` stays default 1 as a documented dead end. The
one hypothesis left standing is the weak variance observation (U=4 probe sd
0.040 vs 0.123, F = 9.5, not established), and it is not worth a run.

**Two findings beyond the hypothesis, both worth keeping:**

1. **The rollout metrics are essentially seed-free.** Across three seeds,
   MSSS and ACC agree to ±0.003 at every horizon — while the SAME heads'
   RAPID k-fold spans 0.245. The instability of the transport probe is not
   in the models and not in the field-prediction skill; it is in the
   projection onto 240 months of RAPID. Field-space evaluation supports
   3-vs-3 comparisons the probe never could.
2. **The model sees months ahead.** Every head beats DAMPED persistence out
   to h=12, and U=1's ACC is still ≈0.40 at a full year. That is the
   programme's first multi-horizon skill statement against the fair
   baseline, and it belongs in the paper regardless of what U it came from.

**Scope note from a code audit (2026-08-11, prompted by Chris: "the U=4
line looks like a bug in training").** What `--unroll U` precisely is
(`temporal.py`, the training loop): after the teacher-forced pass over the
K-window, the window SLIDES — drop the oldest month, append `pred[:, -1:]`,
the model's own prediction of the next month, **with the autograd graph
intact** (no `.detach()`) — the appended token gets the TRUE observation
mask for its month, the next true month is the target, and the extra term
is weighted 1/(u+1). One `backward()` then runs through up to U chained
applications of the shared-weight model: full backpropagation through time.
The audit found **no mechanical bug**: targets align (`zfut[:,u] = Z[t+1+u]`
against a window whose last token represents month t+u), the fed-back-token
masks match what `rollout.py` feeds at eval exactly, the train pool excludes
any window whose unrolled targets would touch holdout months or run off the
array, and U=1 is bit-identical to the plain objective (the loop body never
executes). Two design facts the closure should be read against. First, the
gradient through the un-detached feedback contains an INPUT-SHAPING path —
"make ẑ(t+1) a thing the model maps closer to Z(t+2)" — which under MSE
rewards smoothed, conditional-mean-like predictions; this is the leading
mechanistic explanation for the entire U=4 signature (−29.7% one-step,
probe-SD collapse F = 9.5, lower amplitude ratio, uniformly worse rollout),
and it is exactly the path the PDE-surrogate literature's "pushforward
trick" (Brandstetter et al. 2022) severs by detaching the fed-back state.
Second, the 1/(u+1) weights de-emphasise teacher forcing more than the code
comment implies: at U=4 the self-fed terms sum to 1.083 against the whole
teacher-forced window's 1.0, i.e. ~52% of the objective. **So what E-010 +
E-011 close is the full-BPTT-from-scratch variant.** The detached
(pushforward) variant, and the literature's successful recipe — one-step
pretraining, then a short low-LR unroll FINE-TUNE (GraphCast-style
curriculum) — are untested and would be new experiments, not reruns.

E-013 (this protocol over the 60k E-012 heads) answers whether convergence
changes any of it — for this variant.

---

<a id="e-010"></a>
## E-010 · The unroll question, redesigned around REPLICATES — **COMPLETE: the unroll axis is CLOSED**

**Why this replaces E-009's remaining arms.** Chris, after seeing the
two-arm result: *"reshape toward seeds."*

E-009 asked four unroll values at one seed each. That design can resolve the
**0.28** effect the archive claimed and essentially nothing smaller — 95%
intervals on `rapid_probe_kfold` run about ±0.14. And the 0.28 is exactly
what stopped being believable: #131 (U=1) and #132 (U=2) **rank oppositely**
on the k-fold and on the 36-month split that produced it. So the design was
powered for an effect we no longer think exists, and blind to whatever is
actually there.

**The missing number is the noise floor, and it has never been measured.**
`rapid_probe_kfold` is a day old. The nearest evidence is E-001's two seeds
on the *unpooled head* at 1° (0.690 vs 0.654) and E-003b, where an attempted
second seed reproduced bit-for-bit and was not a seed at all. So when U=1
reads 0.555 and U=2 reads 0.377, **nobody can say whether 0.178 is large.**

**Design.** U ∈ {1, 4} — the two most distant values — at **seeds 0, 1, 2**.
Six runs, 6,000 steps each, everything else pinned to E-009's arms. The seed
travels in `window` as `unroll:4,seed:2`, because `workflow_dispatch` is at
its 25-input ceiling.

**What it answers, in order of importance.**

1. *What is the seed-to-seed spread of the head k-fold?* Three replicates at
   each U give it directly. Every stage-2 comparison in this log has been
   quoted without it.
2. *Do U=1 and U=4 separate above that spread?* If not, **the unroll axis is
   closed** and E-005's result is dead rather than merely withdrawn.
3. Only if they do separate is filling in U=2 and U=8 worth anything.

**What would falsify the unroll hypothesis.** Three seeds at U=1 and three at
U=4 whose ranges overlap. That is the likely outcome on current evidence, and
saying so at dispatch is the point of writing this now.

**Analysis is PAIRED, not two intervals.** `scripts/paired_probe.py` over
arms sharing folds; marginal CIs overlapping is not a test, and the house
rule has said so since E-001.

**Sequencing and a confound to avoid.** The arms must land on ONE box. Boxes
build and cache their own `family3_na025.npz` and two have diverged — #121's
persistence baseline reads 3.139 against 3.343 on the box #131/#132 used, so
a sweep spread across boxes is cross-tensor as well as cross-unroll. Runs go
out one at a time and the runner is checked on each; `sweep_table` reports the
persistence fingerprint and refuses to compare arms that disagree.

E-009's #131/#132 are on the retired box's tensor, so they are **not**
poolable with these six.

**Cost so far.** E-009 spent four dispatches that never started (#135–#138,
~10 s each, wedged runner) plus three arms lost to the disk incident. The
runner `gpu-box-45318655` was dropped from the pool at Chris's instruction to
unblock the queue.

### RESULT, U=1 arm complete (2026-08-11 05:25) — the seed noise is 0.245, and it swallows every unroll effect ever claimed

Three seeds at U=1, one box, one tensor (`adcbe700…`), one codec, 6,000
steps each:

| seed | head k-fold | 95% CI | 36-mo split | z-ratio |
|---|---|---|---|---|
| 0 (#140) | 0.514 | [0.409, 0.636] | 0.524 | 0.472 |
| 1 (#141) | **0.618** | [0.515, 0.709] | 0.476 | 0.472 |
| 2 (#142) | **0.373** | [0.277, 0.557] | 0.505 | 0.469 |
| | mean 0.502, **sd 0.123, range 0.245** | | sd 0.024 | sd 0.002 |

**Set that against the effects this programme has been chasing:**

| claim | size |
|---|---|
| #88 vs #93, the result that started it (36-mo split) | 0.276 |
| E-009's U=1 vs U=2 (head k-fold) | 0.178 |
| **seed range at FIXED U=1 (head k-fold)** | **0.245** |

The noise floor is larger than E-009's unroll gap and the same size as the
original claim. **Nothing about the unroll axis has ever been measured above
it.** E-005's +0.28 was one draw from this distribution against another.

**The instability is in the PROBE, not the model.** The z-ratio — what stage
2 actually optimises — is reproducible to **sd 0.002** across the same three
seeds. So training is stable and the RAPID read-out is not: the same head,
trained to the same forecast skill, projects onto the transport series
anywhere in a 0.245-wide band depending on initialisation. That is a fact
about a 240-month probe with ~9 effective DOF after autocorrelation, not
about the codec or the objective.

**An inversion worth recording, because it complicates the instrument
story.** Under a *tensor* change the split moved 2.7× more than the k-fold
(0.111 vs 0.041); under a *seed* change the k-fold moves 5× more than the
split (0.245 vs 0.048). Both read the same pooled hidden state. Neither is
simply "the noisier one" — they are sensitive to different perturbations, and
the k-fold's advantage is its sample size and its stated interval, not
lower variance under every change. The three k-fold CIs do all overlap, which
is the honest summary: three draws consistent with one distribution.

**What this settles.** Any future stage-2 comparison needs **replicates**, not
arms. A single run's RAPID k-fold carries ±0.12 of seed noise before anything
else is varied, so two configurations differing by less than ~0.25 are
indistinguishable at n=1 — which is every comparison this log has made.

### FINAL RESULT, all six arms (2026-08-11 07:30)

| arm | seed 0 | seed 1 | seed 2 | mean | sd |
|---|---|---|---|---|---|
| **U=1** k-fold | 0.514 | 0.618 | 0.373 | 0.502 | 0.123 |
| **U=4** k-fold | 0.503 | 0.571 | 0.501 | 0.525 | 0.040 |
| **U=1** z-ratio | 0.472 | 0.472 | 0.469 | **0.4710** | 0.0017 |
| **U=4** z-ratio | 0.610 | 0.613 | 0.610 | **0.6110** | 0.0017 |

One box, one tensor (`adcbe700…`), one codec, 6,000 steps, nothing varied but
`U` and the seed.

**1 · Unroll does NOTHING to the AMOC probe.** Difference in means
**+0.023**, SE 0.075, **t = 0.31**. That is one tenth of the U=1 seed range.
Whatever E-005 measured at +0.28, and E-009 at +0.178, it was not this.
**The unroll axis is closed.**

**2 · Unroll makes ONE-STEP forecasting 29.7% worse — and that is the only
horizon we measured.** z-ratio 0.4710 → 0.6110, SE 0.0014, **t = 99**, both
configurations reproducing to sd 0.0017 across seeds. Replicated everywhere
this comparison has ever run: #88 vs #93, E-009's #131/#132, and here at n=3.

**Narrowed 2026-08-11, after Chris asked what the evaluation actually does.**
`z_t+1` is *teacher-forced, horizon 1*: the true 24-month window in, month
t+1 out, for every arm regardless of its training U. That is exactly the
regime a U=4 objective de-emphasises — its loss diverts weight from the t+1
term to self-fed steps — so the direction of this result is close to
expected, and it is NOT a measurement of what unroll exists to buy.
**Multi-step rollout skill, where the model feeds on its own predictions, was
not evaluated in any arm** (`rollout.py` exists for it; no bundle carries a
`rollout.json`). The archive's #88/#93 z-ratios are the same one-step eval,
so this caveat applies to the whole history of the claim.

The honest summary of the axis: unroll costs 29.7% at horizon 1, does nothing
for the transport probe, and its intended benefit is untested. Closing it
fully needs one rollout evaluation over the six E-010 heads — no retraining,
they are all in the artifacts (`temporal.pt`). Until then `--unroll` stays
default-1 on the measured evidence, not on a completed case.

**3 · The probe, not the model, is what is unstable.** Both z-ratio groups
have sd 0.0017 while the k-fold's U=1 group has sd 0.123 — 70× larger, from
the same runs. Optimisation is reproducible; the projection onto 240 months
of RAPID with ~9 effective DOF is not.

**4 · Unroll may STABILISE the probe, and that is the one live hypothesis
left on this axis.** U=1 sd 0.123, U=4 sd 0.040 — a 3.1× reduction. But
F = 9.5 on (2,2) dof, and the 95% critical value is 19.0, so this sits just
past the 90% line and is **not established**. Three seeds give almost no
power to compare variances. It is worth testing properly (5+ seeds per arm)
precisely because it is a different claim from the one that just died: not
"unroll improves the probe" but "unroll makes the probe reproducible". If it
held, it would still have to be weighed against a 29.7% forecast cost.

**5 · Four of the six arms sit BELOW the 0.568 wind-only bar, and both means
do.** U=1 mean 0.502, U=4 mean 0.525, all six 0.513. Only #141 (0.618) clears
it decisively and #147 (0.571) marginally.

This comparison is **direct, not an orientation** — an earlier version of this
entry hedged that it was not, and the hedge was wrong. `probe_kfold` scores
the wind baseline with the same `kfold_r`, on the same deseasonalised RAPID
months, the same year blocks and the same n = 240; only the features differ
(raw τ channels against the head's pooled hidden state). That is exactly what
"does the model beat wind stress" means.

So: **a 6,000-step head does not beat wind stress on this tensor.** Given
E-008 showed 33× more compute does not move this probe either, that is the
uncomfortable pair of facts this programme now has to sit with.

**What this costs the programme.** E-005 is now **dead**, not withdrawn: its
+0.28 was one draw against another from a distribution 0.245 wide. E-009's
two-arm indication goes with it. The honest summary of the unroll line of
work is that it produced one real and reproducible finding — unroll degrades
forecasting — and one artefact that survived four months because nobody had
measured a noise floor.

**The rule that follows.** Stage-2 comparisons need **replicates, not arms**.
A single RAPID k-fold carries ±0.12 of seed noise before anything is varied,
so two configurations differing by less than ~0.25 are indistinguishable at
n=1 — which is every stage-2 comparison in this log before E-010.
## E-009 · The unroll sweep: is U the axis that moves the AMOC probe? — RE-DISPATCHED 2026-08-10 as #131–#134

> **Correction, 2026-08-10 20:1x UTC.** The first version of this entry said
> the sweep was "scored through `probe_kfold.py` … which is the instrument we
> argue from". **That was false, and it made the experiment unable to answer
> its own question.** `probe_kfold` pools the FROZEN EMBEDDINGS along the
> section and fits a ridge; the temporal head is not in it. All four arms
> freeze the same codec, so all four would have returned the same number.
>
> It is not an inference. #116 (frozen codec, 60k head) and #125 (same codec,
> 200k head, different schedule, different optimiser trajectory) return RAPID
> **0.631 [0.513, 0.732], rmse 2.16** — bit-identical, because the only thing
> that differs between them is invisible to that probe.
>
> The original four (#127–#130) were **cancelled while queued**, at no cost,
> and re-dispatched as **#131–#134** on `2b7c3fe`, which adds a year-blocked
> k-fold over the HEAD's own pooled hidden state: same features the 36-month
> split uses, ~240 out-of-fold months, block-bootstrap CI. This is the third
> time this programme has caught a run that was healthy and could not test its
> hypothesis, and the first time the catch came from *two runs agreeing too
> exactly*.
>
> **And it was not new knowledge — it was already in this file.** E-007, on
> #110: *"It must be unchanged, because the codec is byte-identical …
> training the forecaster longer improves forecasting; it does not improve
> what the embedding knows about the AMOC, and those are different columns of
> the master table. Any claim of a 'new best' has to say which."* Exactly
> right, written weeks ago, and then E-005's closing line asked for #88/#93 to
> be *"re-scored through `probe_kfold.py`"* — an impossible instruction that
> propagated into `ml/CLAUDE.md`'s follow-up list and out again into E-009's
> design. Both are struck now. The durable fix is `docs/ML_BASICS.md` §5b:
> the distinction belongs in the document that says what the numbers MEAN, not
> only in the entry of the one experiment that happened to notice it.

**Hypothesis.** At a fixed stage-2 budget, the number of autoregressive
unroll steps `U` in the temporal loss changes the RAPID k-fold correlation
more than the training budget does, and it does so *in the opposite
direction to forecast skill* — a head trained to survive its own errors for
several months learns a slower, more AMOC-like state than one trained to hit
the next month exactly.

**Why now.** The archive says so, from a pair that controls almost
everything. #88 (U=1) and #93 (U=4) share a codec — `z_mse_persistence` is
bit-identical at **3.76004**, which is the strongest evidence available that
the frozen weights are the same file — plus tensor, architecture, seed 0 and
a 6,000-step budget. They differ:

| run | U | z ratio (forecast) | RAPID (36-mo split) |
|---|---|---|---|
| #88 | 1 | **0.494** (better) | 0.173 |
| #93 | 4 | 0.641 (worse) | **0.449** |

Against that, E-007 moved the budget 6,000 → 60,000 and RAPID went
0.319 → 0.321. So on the evidence we have, one axis is worth ~0.28 and the
other ~0.002 — but the two U numbers come from **different dispatches months
apart, scored on the noisy 36-month single split**, and `rapid_r_raw` moves
the other way across the same pair (0.584 → 0.472). That is exactly the
shape of a result that is either the most important thing in the programme
or an artefact of the instrument, and there is no way to tell without
re-measuring all of it at once.

**Design.** U ∈ {1, 2, 4, 8}, everything else pinned: the same frozen codec
(`resume: !run-62,run-63`, 40.693M, verified identical `config` record to
#116/#121/#125), `family3_na025`, K=24, 192×4 head, 6,000 steps, cosine to
the same peak 1e-3, one code version, scored on **`rapid_probe_kfold`** —
the head's pooled hidden state through probe_kfold's year-blocked protocol,
~240 out-of-fold months with a block-bootstrap CI, rather than the 36-month
single split that produced the original pair. U=1 is simultaneously the
sweep's low arm and a re-score of #88 under the modern protocol.

Every arm still writes `probe_kfold.json` too, and every arm's copy will read
0.631. That is not redundancy — it is the control that says the codec was
genuinely held fixed across the sweep, and a divergence there would mean the
arms were not comparable at all.

**What would falsify it.** A flat or non-monotonic k-fold across U, with the
four values inside each other's confidence intervals — that would say the
#88/#93 gap was the 36-month split's noise, and that the +0.28 was never
real. Concretely: if U=1 comes back near 0.45 rather than near 0.17, the
original pair was measuring the split, not the unroll.

**Controls and caveats stated at dispatch.** *n* = 1 per arm, so the sweep
can establish a monotone trend or kill one, and cannot quote a margin: any
gap that survives needs `scripts/paired_probe.py` and a second seed before
it is quotable. The wind-only bar on this tensor is **0.568** and three of
the four archive numbers above sit below it — a sweep that lands entirely
under the bar is a null result about the head, however it orders U.

**Cost.** ~25 min of a 4090 per arm — 6,000 steps is ~3 min at the measured
25.8 ms/step; the rest is the probe ladder. The 95-minute embedding is
**not** in that figure: `Z` for this codec is already on both running boxes'
disks (#125 shows no `embedding` progress records at all, #121 built it with
`"where": "disk"`), so the arms re-use it.

**Sequencing.** All four are queued behind #126, U=1 first: **#131** (U=1),
**#132** (U=2), **#133** (U=4), **#134** (U=8), all on `2b7c3fe`.

The staging rule — Chris, 2026-08-10: *"hold off the second job until the
first job's embedding precomputation is complete"* — exists to stop two jobs
paying the same 95 minutes. Its precondition was checked rather than assumed,
and it is already satisfied: **both online runners hold `Z` for this codec on
disk.** #125's metrics contain no `embedding` progress records at all (it
found the cache), #121's first record reads `"where": "disk"` (it built one),
and the hygiene step is forbidden from pruning embeddings. A queued job can
only be scheduled to an online runner, and the third box is `exited` — so
there is no arrangement of these four runs in which the embedding is computed
twice. Holding them would have bought nothing and cost the fleet an idle GPU
between arms.

Queue order is FIFO, so U=1 still lands first and the remaining three can be
cancelled at zero cost if it turns out to rebuild `Z` after all. That is the
check the monitor is watching for.

Separately, #127 is the first run that can *publish* the cache:
`embed-cache-v1` has **zero assets** because every earlier run carried the
`embed_cache_sync.py` whose exit code was 0 whether or not the upload
happened — so the caller's marker fired and the retry never did. Once it
lands, a box that has never seen this codec pulls 5.2 GiB instead of spending
an hour and a half.

### PARTIAL RESULT — two arms of four, 2026-08-10 22:40 UTC

**Do not quote this as the experiment.** U=4 and U=8 have not run: #133 and
#134 died in `Set up job` when the box filled (see the cost note below), and
a two-point "trend" through overlapping intervals is not one.

| arm | head k-fold, 240 mo | 36-mo split | z-ratio |
|---|---|---|---|
| U=1 (#131) | **0.555** [0.408, 0.685] | 0.413 | 0.484 |
| U=2 (#132) | **0.377** [0.204, 0.528] | 0.465 | 0.569 |

Both completed their full 6,000 steps; #132's failure was in archiving, after
`temporal.json` was written. The control holds: `z_mse_persistence` reads
3.34340500831604 in both, so the codec and the data path were identical.

**The finding that does not need the other two arms: THE TWO INSTRUMENTS
ORDER THESE ARMS OPPOSITELY.** On the year-blocked k-fold U=1 beats U=2 by
0.178. On the 36-month single split — *the instrument #88 and #93 were
compared with* — U=2 beats U=1 by 0.052. Same two runs, same pooled hidden
state, same months; only the resampling differs.

That is the strongest evidence available that the archive's 0.173-vs-0.449
gap was the split rather than the unroll. It does **not** establish the
k-fold's ordering either — n = 1 per arm and the intervals overlap heavily —
but whichever way U=4 and U=8 land, **the original result cannot stand as
measured**, and the E-005 entry that reported it has been amended.

The one thing consistent with the archive: higher U forecasts **worse**
(z-ratio 0.484 → 0.569), as #88 vs #93 also showed.

**Cost, recorded per the house rule.** Three of four arms lost to one
unchecked write: #131's embed-cache push began chunking a 1.5 GiB temp file
onto a box with less free than that, hit ENOSPC, left the part behind, and
took the disk to 50/50. #132 then failed in its archive step and #133/#134 in
`Set up job` — before any step, so the hygiene step that would have cleaned
it could never run. Both halves are fixed (`embed_cache_sync` refuses a chunk
it cannot fit and cleans up in `finally`; `disk_hygiene` clears `Z_*.up`),
and the box needs a stop/start before the sweep can finish.

**Result.** *pending the other two arms.* Read it with

```
node scripts/sweep_table.mjs --runs 131:U=1,132:U=2,133:U=4,134:U=8
```

which tabulates the four arms out of their archived probe bundles beside the
wind bar carried in each bundle, and declines to call any gap significant —
the arms share folds, months and most of their error, so an ordering is all
this design can produce.

---

<a id="e-008"></a>
## E-008 · Is stage 2 COMPUTE-bottlenecked? 60k → 200k — **NO. The forecast keeps improving and the AMOC probe does not move.**

### Result, first arm landed (#125, 2026-08-10 20:02 UTC)

200,000 steps from scratch on the expdecay schedule, GPU, 26.1 ms/step,
1.45 h of training. Archived at
[`ml-metrics/probes-125.json`](https://raw.githubusercontent.com/blauewelt/earth/ml-metrics/probes-125.json).

| stage-2 budget | z-ratio (model/persistence) | RAPID, 36-mo split |
|---|---|---|
| 6,000 | 0.494 | 0.319 |
| 24,000 | 0.406 | — |
| 60,000 | 0.391 | 0.321 |
| **200,000** | **0.383** | **0.317** |

**The pre-registered prediction, checked against what arrived.** The
hypothesis below was written before the run and said two things. RAPID
"should not move": 0.321 → **0.317**, confirmed. And z-ratio "should land
near 0.36–0.37": it landed at **0.383**, *outside* the stated band. The
extrapolation was too optimistic — the curve is decelerating faster than its
own first two differences implied. Recording the miss rather than widening
the band retrospectively is the point of writing it down first.

**What this closes.** From 6,000 to 200,000 steps — 33× the compute — the
forecast objective improves steadily (0.494 → 0.383, and it has not stopped)
while the AMOC probe sits inside ±0.004 of where it started. The head gets
monotonically better at predicting **z** and no better at all at predicting
**transport**. Stage 2 is not compute-bottlenecked; more precisely, the thing
more compute buys is not the thing we want.

That leaves the parameter-bottleneck arm (b), which needs a from-scratch run
at larger width, and the objective — which is why E-006 exists.

**A caveat that is not small.** All four RAPID figures above are the
**36-month single split**, because until today that was the only instrument
that could see stage 2 at all (see the note in E-009: `probe_kfold` scores the
codec, and returns 0.631 for every one of these runs). Its standard error is
of order 0.15, so "flat to ±0.004" across three points is a good deal tidier
than the instrument can honestly resolve, and the agreement is partly luck.
The k-fold over head features — 240 out-of-fold months, block-bootstrap CI —
lands with #131–#134 and is what these numbers should be re-read against.

---

### #121 is NOT a third schedule arm — it is on a different tensor

Landed 02:24 on 2026-08-11: cosine, 200,000 steps, CPU, on
`gpu-box-35586926`. It was dispatched as the cosine comparison against #125's
expdecay, and it cannot serve as one.

Its persistence baseline reads **3.139432907104492** where #125, #126, #131
and #132 all read **3.34340500831604** — 6.1% apart in a quantity that is
data-only (`x_t` against `x_{t+1}` on held-out months) and therefore cannot
depend on the model. Its codec k-fold reads 0.627 against their 0.631.

Both numbers reproduce the confound recorded in E-007: **the baseline tracks
the BOX, not the run.** #121 ran on `gpu-box-35586926`, the others on
`gpu-box-45318655`, and the two boxes build and cache their own
`family3_na025.npz`, which have diverged. So #121's 36-month RAPID of 0.498
against #125/#126's 0.317/0.318 is a cross-tensor difference with a schedule
change inside it, and nothing can be attributed to the schedule.

What is unaffected: **#125 vs #126** share a box and agree to sixteen digits,
so the decay-to-zero result below stands.

This was caught by `scripts/sweep_table.mjs`'s control check on its first real
use — which is the argument for building the control before you need it. The
sha256 that exists to answer this question directly (E-007's provenance fix)
was shipping only in the checkpoint artifact, which the probe bundle never
reads; it now rides with the probe results.

**Consequence for E-009:** the remaining arms must run on the SAME box as
#131 and #132, or the sweep is cross-tensor as well as cross-unroll.

### The box effect, finally MEASURED — 0.041 on the head k-fold

Two things landed on 2026-08-11 that turn this from a warning into a number.

**The tensors differ, by hash.** `provenance.json` now ships with the probe
bundle, so:

| box | tensor sha256 | z-persistence |
|---|---|---|
| `gpu-box-47094145` | `b40f5b0b253005cb…` | — |
| `gpu-box-35586926` | `adcbe700fb6e160b…` | 3.139432907104492 |
| `gpu-box-45318655` (retired) | — | 3.34340500831604 |

No more inferring divergence from an anomaly in a baseline: the hashes simply
disagree.

**And #140 is a controlled replicate of #131.** Same U=1, same seed 0, same
codec (weight hash `6c52f0687b`, verified on both boxes), same everything —
differing only in which box, and therefore which tensor:

| | head k-fold | 36-mo split |
|---|---|---|
| #131, tensor 3.343 | 0.555 | 0.413 |
| #140, tensor `adcbe700` | **0.514** | **0.524** |

**The tensor alone moves the head k-fold by 0.041 and the single split by
0.111.** For scale, E-009's U=1-vs-U=2 gap was 0.178. So the box effect is
about a quarter of the effect that experiment was trying to measure — small
enough to have hidden, large enough to matter, and exactly the reason
cross-box arms cannot be pooled.

It also says something about the two instruments: the split moved 2.7× more
than the k-fold under the same perturbation, which is consistent with it
being the noisier read-out and is a second, independent reason to prefer the
k-fold.

**The root cause is fixable and now fixed.** Boxes built their own tensors
because there was no shared artefact to pull — the embed cache had never
published, and the reason was `curl --data-binary`, which buffers the whole
body and died on the first 1.5 GiB chunk every time. With `-T` streaming, a
published Z means every box trains stage 2 on identical embeddings and this
whole class of confound goes away.

---

### E-008e · Does decaying the LR to ZERO actually help? — YES, BY 0.16%

Chris, 2026-08-10: *"I'm not sure I agree with the literature that decaying to
0 is beneficial, is this double verified? maybe we can verify with an extra
run."* So: #126, identical to #125 in every input, differing only in whether
the schedule's tail reaches zero.

**It is very nearly a paired experiment, which is why n = 1 is worth
something here.** The two runs share seed 0, the same box, the same frozen
codec and the same tensor — the channel-space persistence baseline reads
**3.34340500831604 in both, to sixteen digits** — and the schedules are
*identical until step 180,000*:

| step | LR #125 | LR #126 | z-MSE #125 | z-MSE #126 |
|---|---|---|---|---|
| 2,000 | 1.000e-3 | 1.000e-3 | 1.98694 | 1.98694 |
| 100,000 | 1.830e-4 | 1.830e-4 | 0.83738 | 0.83738 |
| 180,000 | 4.575e-5 | 4.575e-5 | 0.81191 | 0.81191 |
| 190,000 | 3.847e-5 | 1.923e-5 | 0.78504 | **0.78369** |
| 196,000 | 3.467e-5 | 6.933e-6 | 0.78491 | **0.78255** |
| 198,000 | 3.349e-5 | 3.348e-6 | 0.75719 | **0.75494** |
| 200,000 | 3.235e-5 | **0.000e+0** | 0.79618 | **0.79379** |

Bit-identical for the first 90% of training, so there is no seed noise in the
comparison at all: the only difference is the last 10% of the schedule, and
**every one of the five logged points after the divergence favours the
to-zero run.**

**The result.** On the full evaluation, not a training batch:

| | z-ratio | channel ratio | RAPID (36-mo) |
|---|---|---|---|
| #125, LR floors at 3.2e-5 | 0.38251 | 0.40881 | 0.317 |
| #126, LR reaches 0 | **0.38190** | **0.40763** | 0.318 |
| gain | **0.159%** | **0.288%** | — |

**So the literature is right in direction and negligible in magnitude.**
Decaying to zero helps, reproducibly and consistently across every point
after the schedules part, by about two parts in a thousand on the objective
stage 2 actually optimises. The AMOC probe moved 0.001, which on a 36-month
split is nothing.

Note what #125 was, precisely: not "no decay" but a floor at 3.2e-5, i.e.
3.2% of peak. This measures the last 3% of the ramp, not decay versus
constant.

**What it settles for us.** Nothing about the schedule choice is worth
another run. `expdecay` with a terminal taper is horizon-free, which is the
property that actually mattered, and the taper's endpoint is worth 0.16% —
so pick it for the horizon-freedom and stop thinking about the tail. Chris's
scepticism was well placed: the effect is real, and it is not a reason to do
anything differently.

---

### The dispatch record

**Run** #120 · `head_sha` 502364279 · started 14:59 UTC ·
`window: warm2:f3_s2_60k__temporal@1e-4`, `temporal_steps: 140000` (the EXTRA,
not the total), `job_timeout: 1400`, `resume: !run-62,run-63`.

**Three dead dispatches before it.** #117 and #118 were infrastructure; #119
was the design being wrong, which is the more useful of the two. #117 (11:19) cleared
the disk guard with ~11 GB free and then spent an hour memmapping a **10.4 GiB**
embedding cache onto a disk with 5 GiB left; it was cancelled at 45/50 GB
rather than allowed to fill the disk, which would also have taken the runner
offline. #118 (12:40) sat queued against an idle runner — the documented queue
stall — and its re-dispatch as #119 needed a box stop/start before it was
picked up. #119 then embedded for 93 minutes and refused at the resume, for
the reason below. None of the three says anything about the hypothesis; all
three are recorded because the log is how the COST of an experiment is known
as well as its result, and E-008 has so far spent about five and a half hours
of wall clock and roughly two and a half hours of GPU without producing a
single training step.

**Note on the measurement chain.** #119 embedded in **float32, in RAM** (the
disk could not hold a 10.4 GiB cache, so `_cache_plan` chose memory and wrote
nothing). #120 caches in **float16** at 5.2 GiB, which fits, and publishes it
to the `embed-cache-v1` release so no later run repeats the 95 minutes. The
dtype shifts the model/persistence ratio by ~2e-7 — seven orders below the
effect — but it is a change to the chain, and the place for that is the log
rather than a diff someone finds later.

**The question.** Chris asked it precisely: stage 2 is either (a) compute
bottlenecked — 140k more steps help — or (b) parameter bottlenecked, which a
checkpoint cannot answer and which needs a from-scratch run at larger width.
This tests (a) only. (b) is deliberately deferred.

**Hypothesis, written before the result.** E-007's three converged points
(6k → 0.494, 24k → 0.406, 60k → 0.391 z-ratio) are still improving and
decelerating: −0.103 then −0.052 per stage. Extrapolating that deceleration,
200k should land near 0.36–0.37 in z-ratio, and — this is the part that
matters — the **RAPID probe should not move**, because it already plateaued
between 24k and 60k (0.319 → 0.321) while the forecast loss kept falling.
The falsifier is symmetric and worth stating: if RAPID rises past ~0.35, the
plateau was a compute artefact and E-007's conclusion is wrong; if z-ratio
also stops falling, stage 2 is converged and neither more compute nor this
architecture will help.

**This is a WARM RESTART, not a fourth point on E-007's curve.** Every E-007
point ran its own cosine from peak to zero over its own budget, so the three
are three converged models. This run instead takes the 60k head's **weights**
and trains 140,000 more steps on a fresh cosine at one tenth the original peak
(1e-3 → 1e-4). It is comparable to E-007 as an endpoint and **not** comparable
as a trajectory.

**Correction (2026-08-10, after #119).** The first version of this entry said
the run "reloads the 60k model, optimiser moments and RNG stream". That was
false, and #119 is how it was found out: `--resume-temporal` refused after
93 minutes of embedding, with

> `f3_s2_60k__temporal.pt` predates optimiser-state saving (missing
> `['opt', 'sched', 'step']`) … a warm restart wearing a continuation's name.
> Refusing.

Checked against the release afterwards: `f3_s2_60k__temporal.pt`,
`f3_s2_24k__temporal.pt` and every rescue mirror are `{args, model}` — **no
published head can be continued at all.** The optimiser-state mirror landed on
2026-08-10 and no stage-2 run has succeeded since, so the first continuable
head will be the one this experiment produces.

The guard did its job: it refused rather than silently reset Adam's moments and
report the result as a continuation. But note what it cost — the refusal fires
*after* the embedding, at the point of use, when it could have fired at the
point of dispatch. The checkpoint's contents are knowable in the first ten
seconds of the job.

So E-008 runs as an explicit warm restart: `--init-temporal
f3_s2_60k__temporal --steps 140000 --lr 1e-4`, where `--steps` is the EXTRA,
not the total. It logs `stage2_warm_restart` rather than `stage2_resumed`, and
names what was reset (moments, schedule, RNG), so no later reader has to take
this paragraph's word for it. `tests/test_resume_temporal.py` pins that a warm
restart both trains and lands somewhere other than a straight-through run —
which is the whole reason the two get different flags.

Planned LR, for checking against the live series — a FRESH cosine over
140,000 at peak 1e-4, so it starts at the peak rather than partway down one:
0 (total 60,000) → **1.000e-04**; 20,000 → 9.505e-05; 35,000 → 8.536e-05; 70,000 → 5.000e-05; 105,000 → 1.464e-05; 140,000 → 0.000e+00. (An earlier version of this paragraph listed 7.939e-05 at step 60,000,
which was the continuation schedule this run turned out not to be able to use.
It is replaced rather than annotated, because a stale reference table is worse
than none: it is checked, it matches nothing, and the reader blames the run.)

**The LR bug this design was originally built around, kept because it is
still live on the resume path.** `CosineAnnealingLR.load_state_dict` restores
`T_max` and `base_lrs` from the parent, so a reloaded schedule asked for a
larger total believes it already finished and returns **lr = 0.0**: hours of
"continuation" that change nothing while every status reads success. That was
live in `temporal.py` and was caught only because Chris asked for a lower LR.
`tests/test_resume_temporal.py::test_extend` pins it and the trainer refuses
to start at a non-positive LR — which will matter the moment there is a
checkpoint that CAN be continued, i.e. the one this run produces.

**Confound to carry forward.** The parent (#112) trained on run-62's tensor.
`resume: !run-62,run-63` requires the same one, and provenance now records a
sha256 of the tensor file, so a repeat of the #88-vs-#110 box divergence would
be visible rather than inferred.

**Status:** training. Nothing to conclude yet.

---

<a id="e-003b"></a>
## E-003b · The "second seed" that was not one — #116 reproduced the head probe bit-for-bit

**Run** #116 · `head_sha` 23b99002d91268e02c0ad601b74bb1218137e005 · frozen 40M
anchor, no training, every eval on the GPU.

**What it was for.** Two things: move `dip_check.py`, `rollout.py` and the
standalone `trainprobe.py` onto the GPU (they had all been embedding a 40.7M
codec on CPU, which is what made #112's tail burn hours at `gpu_util = 0`),
and draw a **second seed** for the unpooled attention head so that 0.662 would
stop being a single-seed number.

**The first half worked. The second half was impossible, and the run reported
success anyway.**

| probe | this run | previously reported |
|---|---|---|
| unpooled attention head | 0.662 · [0.557, 0.745] · 2.10 Sv | 0.662 · [0.557, 0.745] · 2.10 Sv |
| raw-3×3 control | 0.628 · [0.514, 0.729] · 2.17 Sv | 0.628 · [0.514, 0.729] · 2.17 Sv |
| pooled ridge (k-fold) | 0.631 · [0.513, 0.732] | 0.631 · [0.513, 0.732] |

Bit-identical, to every digit printed. The cause is that `probe_head.py` had
**no seed argument at all** — its three per-fold seeds were the literal tuple
`(0, 1, 2)`, averaged, and the file name carried no seed either. So the
dispatch labelled "seed B" recomputed a deterministic estimator and would have
overwritten seed A with itself. This is CLAUDE.md §6c rule 7 in its purest
form: the run asserted the invocation, and nothing asserted the effect.

**What it does establish**, and it is not nothing: the whole ladder reproduces
exactly across runs and boxes, which — after the #88-vs-#110 tensor divergence
— is worth having measured. The ridge reproducing 0.631 confirms #117's parent
tensor is the one we think it is.

**What it does not establish:** that 0.662 is robust to resampling. It remains
one estimator, and `head − raw3×3 = +0.034` remains unquotable.

**Fixes landed with this entry.** `probe_head.py --seed-base N` makes the
seeds `(N, N+1, N+2)` and puts `N` in the filename, so two draws cannot
overwrite each other. More usefully, the probe now dumps its **out-of-fold
predictions, target and year blocks**, and `scripts/paired_probe.py` scores
the two probes' difference by resampling YEARS and rescoring both on the same
resampled years. That is the right instrument and a second seed never was:
the head's CI [0.557, 0.745] and the control's [0.514, 0.729] overlap almost
entirely, but they share their folds, their months and most of their error,
and shared variation cancels in a paired difference. Whether +0.034 survives
that is now a question we can answer for the price of one CPU-minute instead
of a GPU-hour, and `tests/test_paired_probe.py` pins that the script refuses
an imaginary gap as readily as it resolves a real one.

**Second bug, found in the same artefacts and fixed.** `probe_sequence.json`
came back all-NaN across the whole K sweep — again, as it had on #101, where
it was logged as "the sequence probe has a bug of its own". It was not a probe
bug. `probe_sequence.py` selected the 26.5°N section with
`isfinite(d["X"][0, sec_y, :, 0])` — **month 0, channel 0**. Family-3's
channel 0 is `cur_speed` (GLORYS, from 1993-01) and the tensor starts 1982-01,
so channel 0 is NaN everywhere in month 0, the section came out **empty**, and
`z.mean(0)` over zero rows is NaN — for every month, for every K. The
seasonal-only floor stayed finite (−0.168 / 0.272) precisely because it never
touches the embedding, which is what made the failure read as a broken probe
rather than a broken mask. It now uses the same any-month mask `temporal.py`
and `probe_head.py` use, so the sequence probe scores the same 265 pixels as
the rest of the ladder, and it **exits** rather than writing NaN to a results
file. `tests/test_section_mask.py` pins all of it, including that the old rule
and the new one agree whenever no channel starts late — the old rule was not
wrong in general, it was wrong for this tensor.

**Cost of the misdiagnosis:** one dispatch, and #101's K-sweep, which has been
uninterpretable since 2026-08-09 for a reason that was in the channel list the
whole time.

---

<a id="e-007"></a>
## E-007 · How far past persistence can a FROZEN-codec forecaster go? — still improving at 60k, but the AMOC probe plateaus at 24k

**Question, from Chris:** before redesigning the loss, how much better than
persistence can the existing pipeline get simply by training the stage-2 head
longer? If it saturates early, the forecaster was never the bottleneck and the
whole joint-training premise is misdirected.

**Design.** Stage 2 only — the codec is frozen at the 40M anchor, so no
degeneracy is available and nothing about the embedding can move. U=1, K=24,
head 192x4. Each budget is its own *converged* endpoint, because
`temporal.py` anneals `CosineAnnealingLR` over `a.steps`; two budgets are two
converged runs, not an early and a late snapshot of one.

| run | steps | z-space ratio | **data-space ratio** | **vs persistence** | RAPID r (deseas.)* |
|---|---|---|---|---|---|
| #88 | 6,000 | 0.494 | 0.576 | +42.4% | 0.173 |
| #110 | 24,000 | 0.406 | 0.473 | +52.7% | 0.319 |
| **#112** | **60,000** | **0.391** | **0.420** | **+58.0%** | **0.321** |

\* single 36-month blocked split from the temporal hidden state, n_test = 36 —
noisy, and NOT the k-fold the programme argues from.

**Answer: still improving, but decelerating — and the two metrics part ways.**
6k → 24k bought −0.103 in the data-space ratio; 24k → 60k bought −0.052 for 2.5×
the steps. Forecast skill has not plateaued. The RAPID correlation HAS: 0.319 →
0.321 is nothing. So past 24,000 steps the model keeps becoming a better field
forecaster without becoming a better AMOC probe, which is the more interesting
half of the result — the two objectives stop moving together.

6,000 steps — the budget every previous stage-2 run in this programme used —
was leaving a great deal on the table, so some of what has been attributed to
codec quality may have been an undertrained forecaster.

**#112 was cancelled at 09:00 with its result already computed.** Its GPU had
been idle for 25+ minutes at 60% CPU — the 60,000 training steps finished
around 07:00 and only the CPU probe tail was left, against a 09:51 timeout.
`temporal.py` writes `stage2_result` at the END of its own run, so the number
existed on disk hours before anything would have published it. Weights and
metrics were both recovered by a rescue-only dispatch and are on the release as
`f3_s2_60k__temporal.pt`. See docs/INFRASTRUCTURE.md §2b.

**What did NOT change: the headline.** #110's pooled year-blocked k-fold RAPID
is **0.627 [0.503, 0.735]** against the anchor's 0.631 [0.513, 0.732]. It must
be unchanged, because the codec is byte-identical — and it is. Training the
forecaster longer improves *forecasting*; it does not improve what the
*embedding* knows about the AMOC, and those are different columns of the
master table. Any claim of a "new best" has to say which.

**Confound to note.** `--resume "!run-62,run-63"` takes whichever checkpoint
the box happens to hold: #88 got run-63, #110 and #112 got run-62. Different
codecs, hence different persistence baselines (channel-space 1.205 vs 1.154).
Each run divides by its own persistence so the ratio is within-codec and the
comparison is largely fair, but **#110 vs #112 is the clean pair**; #88 is
indicative.

**A confound that is NOT float noise.** The channel-space persistence baseline
is data-only — x_t against x_{t+1} on held-out months — so it cannot depend on
the model and must be identical for every run on the same tensor. It is not:

| run | box | chan persistence baseline |
|---|---|---|
| #88 (6k) | gpu-box-45318655 | 1.2046650648117065 |
| #110 (24k) | gpu-box-35586926 | **1.1540812253952026** |
| #112 (60k) | gpu-box-45318655 | 1.2046650648117065 |

It tracks the **box**, not the run: same box gives bit-identical values to 16
digits, a different box is **4.2% away** — about 350,000× float32 epsilon, so
this is not precision. Each box builds and caches its own
`ml/cache/family3_na025.npz`, and two of them have diverged. Every cross-box
comparison in this programme is therefore uncontrolled to some degree, and
nothing in any run's output said so.

Bounded impact here: rescoring #110 on the other box's baseline moves it from
0.473 (+52.7%) to 0.453 (+54.7%), which if anything strengthens the trend and
leaves the 6k → 24k → 60k ordering untouched. The k-fold RAPID numbers agree
across boxes too (0.627 vs 0.631), so the tensors are close, not unrelated.
**Fix:** every run's `provenance.json` now records a sha256 of the tensor it
actually used, so divergence is visible instead of inferred from an anomaly in
a baseline nobody was looking at.

**Consequence for E-006.** The bar the data-space loss has to clear is **0.420**,
not the 0.576 that a programme-wide habit of 6,000-step stage-2 runs made feel
normal. A redesign that merely matched the old number would have looked like a
win against a baseline that was simply undertrained.

---

<a id="e-004"></a>
## E-004 v5 · The scale-free ratio has a SECOND degeneracy: inflate the baseline — #107/#108 void

The live-persistence denominator did close the shrinkage direction. The
encoder went the other way instead.

`l_pers = ‖z_t − z_{t+1}‖²` is the persistence baseline **in z-space**, so it
is not only a scale — it is *how hard the forecasting problem is*, and the
encoder writes it. `r_fore = (l_fore/l_pers)/ref` falls just as happily by
making the denominator BIG as by making the numerator small. Add a large,
temporally structured component to z that persistence cannot track but the
head can — the head is handed sin/cos of month-of-year, so a strong seasonal
oscillation is the cheapest such component — and the ratio collapses without
any forecasting having improved.

| | step 120 | step ~4000 | z-scale vs step 0 | l_fore/l_pers |
|---|---|---|---|---|
| #107 `lse@0.44` | l_pers 22.8 | l_pers 311 | **77× larger** | 0.545 → **0.025** |
| #108 `sum@0.44` | l_pers 10.7 | l_pers 1043 | **250× larger** | 0.584 → **0.002** |
| #101 frozen (control) | l_pers 4.22 | l_pers 4.10 | 1.0× | 0.727 → 0.426 |

In amplitude terms the month-to-month change in z grew **8.6×** (#107) and
**15.8×** (#108). The resulting forecast numbers read "explains 97.5% / 99.8%
of what persistence leaves", against the frozen codec's 56% — which is the
tell. A jump like that is not a geophysical forecasting result, it is a
bookkeeping one. `r_rec_probe` stayed at 0.73–0.82 and 0.99–1.11 respectively,
so reconstruction never objected: the decoder can simply subtract a component
it knows about.

As before, `sum` drove it harder than `lse` (250× vs 77×) for the same reason
as last time — the sum always weights the forecast term, the smooth max only
when it is the worse one.

**I had named this failure and ranked it unlikely.** While designing the
live-denominator fix I wrote, in my own notes, that the model "could make
z_{t+1} − z_t bigger while keeping the prediction error the same — a real
possible cheat but far less trivial than pure scaling. Worth noting, not worth
blocking." It was not less trivial. It was the dominant direction, and it
arrived faster than the shrinkage it replaced. **A degeneracy you can name is
one you must close or measure — never one you may rank as improbable.**

**And the detector I built for the first degeneracy was one-sided.**
`z_shrink` logged the failure correctly (0.003) but `status.html` only turned
it red above 1.2, so a 250× *expansion* rendered in grey as "0.00x" and looked
ordinary. That is the exact principle written into CLAUDE.md §6c twenty
minutes earlier — *what would look identical whether this works or fails?* —
violated inside the instrument written to enforce it. The guard is now
two-sided on |log₂(scale)|.

**Verdict.** #107 and #108 are void; both cancelled mid-run. No forecast
number from either is reportable, and the `lse` vs `sum` comparison they were
meant to settle is not settled — both were optimising the same artefact, so
their difference measures only which loss shape exploits it faster.

**This is the fourth failure of the same slot** — step-0 constant, detached
ratio, hand-copied constant, and now baseline inflation — and every one of
them comes from scoring the forecast in a space the model authors. E-006
below is the fix, and this entry is the argument for building it rather than a
fifth normalisation.

---

<a id="e-006"></a>
## E-006 · The loss, rewritten in the data's units — BUILT 2026-08-10, not yet run

Chris, after four failed normalisations: *"the loss term should just be (1) how
much have we failed to predict X + (2) how much have we failed to predict Y.
that's it. and alignment of the 'failure' on roughly the same scale can't be
too hard."*

He is right, and the reason the alignment kept being hard is the thing worth
writing down: **we were measuring one of the two failures in a space the model
invents.** Reconstruction failure is measured against observed channels —
fixed, external, ungameable. Forecast failure was measured against `z`, which
the encoder is free to rescale, so its "scale" was never a scale at all. Every
denominator we tried was an attempt to referee a quantity that had no fixed
units. Four attempts, four failures, all downstream of that one choice.

**The design.** Decode the forecast back into the data before scoring it. The
temporal head predicts `ẑ_{t+1}`; push it through the SAME decoder
reconstruction already uses and score it against the observed channels at
t+1. Then both terms are "failed to predict real, standardised observations",
and the sum needs no referee:

    L = MSE(x̂_t^masked, x_t) / var(x)  +  MSE(x̂_{t+1}, x_{t+1}) / var(x)

Both denominators are variances of the OBSERVED field, computed once from the
tensor. They are constants — but constants *of the data*, which is exactly the
distinction that matters: a constant denominator is fine when the model cannot
move it, and poison when it can. 1.0 means "no better than climatology" for
both terms, in the same units, with no control run anywhere.

**What this buys, beyond simplicity.**

- **The shrinkage degeneracy cannot exist.** It is not closed, or policed, or
  penalised — there is no free direction to close. Shrinking z shrinks the
  decoded field too, and the target is a real observation that does not move.
- **No frozen-codec reference, so no sequencing, no hand-copied constant, and
  no dependence on how long a control happened to run.** The twin head of
  E-004d becomes unnecessary; it was an elegant answer to a question that
  should not have been asked.
- **`sum` versus smooth-max stops being load-bearing.** The max existed only
  because the two terms were incommensurable and a sum would have been
  dominated by whichever was bigger. Once both are fractions of the same
  variance, a plain sum does what Chris asked for originally — if either
  failure is high, the total is high — and it is the version you can explain
  in one line. Keep `lse` behind a flag for comparison; do not lead with it.
- **The objective becomes one sentence**: reconstruct the present you were
  shown, and the future you were not.

**The algebra, checked before the code** (`tests/test_e006_algebra.py`,
2026-08-10). CLAUDE.md 6c rule 5 asks for this and the four retracted
normalisations are why. Let `s` be the encoder's free output scale — the knob
the model actually found, twice: `z_shrink` reached 1/40 in one run and ×250
in another.

| loss | `L(s)` | `dL/ds` | meaning |
|---|---|---|---|
| z-space (E-004 family) | `s²‖a−b‖²/c` | `2s‖a−b‖²/c` | strictly increasing, so descent shrinks z — **the cheat** |
| data space (E-006) | `‖aw−b‖²/var(x)` | **exactly 0** | a gauge: nothing to police |

The second row is the design in one line. Under the reparametrisation
`z → s·z`, `decoder → decoder/s` the decoded field is unchanged, so the loss
cannot see `s` at all. The shrinkage degeneracy is not closed, penalised or
guarded — **there is no free direction for it to live in**. Also checked: the
persistence baseline has `dL/d(pers) = 0` because it is not in the objective
at all (it stays a logged diagnostic, rule 3), and `∂var(x)/∂θ = 0` for every
model parameter θ, which is what makes a constant denominator legitimate here
and poison in E-004. Finally `∂L/∂rec = ∂L/∂fore = 1/var(x)`, which is why a
plain sum replaces the smooth max: neither term is privileged by a weight
anyone chose.

**The honest costs.**

- One decoder pass per step, which is cheap next to the K gradient-carrying
  encoder passes already being paid.
- The forecast term is harder than the masked-reconstruction term, so it will
  sit higher (order 0.7 vs 0.2). That is a real difference in difficulty, not
  an artefact, and a sum weights them by their gradients rather than by a
  number we chose.
- We lose the read-out that said "as good as the codec we started with"
  directly in the loss. That was a DIAGNOSTIC living in the objective, which
  is what caused the trouble; it moves to the logs, where `r_rec_probe` and
  `z_shrink` already are.
- Per-channel variances, not one global one, or the loss is dominated by
  whichever channel has the largest anomaly variance.

**Status — BUILT 2026-08-10, not yet dispatched.** `--loss-mode data` in
`ml/train_joint.py`. The condition set here ("nothing is dispatched against
this until the algebra and a synthetic smoke test are both done") is now met:

| check | file | what it establishes |
|---|---|---|
| algebra, symbolic | `tests/test_e006_algebra.py` | `dL/ds = 0` under the gauge; the baseline is absent from the objective; `∂var(x)/∂θ = 0` |
| gauge, **on the real model** | `tests/test_e006_gauge.py` | scaling `to_z` by 4 and the decoder's z-columns by ¼ leaves the decoded field identical to 2e-5, so the data-space loss is unchanged (ratio 1.000000) while the z-space loss moves by exactly s² = 16.000000 |
| end to end | `tests/test_e006_smoke.py` | 30 months × 8×10 × 5 channels, 2-layer codec, CPU, ~1 min; the run completes, logs no NaN, and the two terms land **1.2–1.3× apart** |

That last number is the formulation's whole claim, and it is worth stating
beside the thing it replaces: under the reference normalisation the two terms
sat at ~1.0 and ~0.3 through *different* denominators, so the smooth max was
always reconstruction and 95.7% of every gradient went to it (measured on
#94). Here they are fractions of the same observed variance and a plain sum
sees both.

Three implementation details that are decisions, not transcription:

- **The reconstruction term is divided by `var_c` too.** It has to be, or the
  sum adds a per-channel-normalised forecast to an unnormalised
  reconstruction. This does change stage 1's objective slightly relative to
  `train.py` — the codec's own pretraining is not per-channel normalised —
  and that is deliberate: within this loss the two terms must share a unit.
- **`var_c` is computed before the NaNs are zeroed**, from finite entries of
  the training years and training longitudes only. Zeroing first would mix
  land into the variance of every coastal channel.
- **`--ref-fore` and `--lam` are refused** rather than ignored. Both ask for
  a referee between the two terms, and accepting either would produce a
  healthy-looking run that cannot test its own hypothesis.

Still to decide before dispatch: the budget, and whether the control is the
frozen codec (`#116`'s 0.631) or a `--loss-mode lse` run of the same length.
It should be the frozen codec — the point is whether joint training beats not
doing it at all.

---

<a id="e-004-2"></a>
## E-004 · The forecast term was rewarding the encoder for SHRINKING z — every joint result so far retracted

Caught live on 2026-08-09 at 21:47 UTC, on the two control runs launched to
calibrate the fix from the previous entry. It is the largest error in this log
and it invalidates the forecast side of every joint run: #91, #92, #94, #95,
#100, #102.

**The mechanism.** `l_fore` is a mean squared error *in z-space*. It was
divided by `ref_fore`, a constant measured once at step 0. Nothing in that
arrangement asks the encoder to forecast better — it asks for a smaller
numerator, and the cheapest way to get one is to make ‖z‖ smaller. The decoder
simply rescales, so reconstruction barely notices: a free direction, straight
down.

**The evidence.** `l_pers` — the persistence loss ‖z_{t+1} − z_t‖², which
depends on the codec alone and not at all on the forecaster — is the ruler.
Frozen codec, it should not move. Training codec, it measures the contraction.

| step | #101 frozen: l_pers | #102 training: l_pers | #102 contraction | #102 `r_fore` (as logged) | #102 `l_fore/l_pers` (scale-free) |
|---|---|---|---|---|---|
| 120 | 4.221 | 0.533 | 7.8× | 0.353 | 0.802 |
| 480 | — | 0.227 | 18.4× | 0.138 | 0.736 |
| 960 | — | 0.112 | 37.2× | 0.068 | 0.738 |
| 1200 | — | 0.103 | **40.5×** | **0.054** | **0.636** |
| 1680 | 4.103 (1.0×) | — | — | — | (#101: **0.595**) |

Read the last two columns together. The logged forecast number fell by 6.5×
and looked like a triumph. The scale-free ratio moved 0.80 → 0.64, and the
**frozen** codec — no stage-1 training whatsoever — was already at 0.595. Once
the contraction is divided out, joint training was not ahead of the frozen
codec. It was behind.

`r_rec_probe` stayed between 1.04 and 1.18 throughout, which is exactly why
nothing caught it: the guard that was supposed to notice the codec paying for
the forecast was watching a quantity the cheat does not touch.

**This also retracts the `--ref-fore 0.29` of the previous entry.** That number
came from #95's tail `r_fore`, so it was mostly contraction — and #95 was a
*treatment* run, not a control. A circular reference calibrated on an artefact.
The honest reference is a frozen-codec control's converged `l_fore/l_pers`,
which #101 puts near **0.58**.

**The fix, and why it is not a tripwire.** Divide the forecast loss by *the
batch's own* persistence. Shrink z and numerator and denominator shrink
together, so the direction pays nothing — the degeneracy is closed by
construction rather than policed. `--ref-fore` then scales that ratio onto
`r_rec`'s footing so 1.0 means "as good as the frozen-codec pipeline" for both
terms and the smooth max compares like with like. Chris's rule is untouched:
no threshold, no penalty, no job killed. The loss simply cannot be cheated this
way. `z_shrink` is now logged every step and shown on the status panel, because
a run that improves by contracting is indistinguishable from one that learns
unless you plot it.

**Standing lesson.** A normalised loss is only as honest as its denominator.
When the numerator is a norm in a space the model itself defines, a *constant*
denominator is an invitation to rescale that space. Normalise against a
quantity computed in the same space at the same moment, or the ratio measures
the units rather than the skill.

**The first fix was wrong, in an instructive way.** #103/#104 shipped
`l_fore / l_pers.detach()`. Detaching the denominator makes it a constant
*with respect to the parameters inside the step*, which is exactly the
condition that pays for shrinking z — the ratio only looks scale-free, its
gradient is not. Both losses are quadratic in z, so under a rescale z → a·z:

| denominator | r_fore(a) | dr/da at a=1 |
|---|---|---|
| detached | a²·l_fore / const | **+2·l_fore/l_pers** — shrinking lowers the loss |
| live | a²·l_fore / (a²·l_pers) | **0** — rescaling buys exactly nothing |

Verified numerically on a toy tensor before relaunching: +1.81 detached,
+2.5e-7 live. #103 reached **1099× contraction by step 600** — worse and far
faster than #102, which is what the algebra predicts: making the forecast term
matter without removing the incentive means the smooth max now points the
codec down the free direction with real weight. `r_rec_probe` climbed to 1.94
alongside, so the codec was visibly being destroyed; the `z_shrink` series
added in the same commit is what made it a five-minute diagnosis instead of a
retracted result three hours later.

**The reference, finally measured.** #101 ran 12,000 steps with the codec at
learning rate 1e-12. Two pins say it did what it claimed: `r_rec_probe` read
**exactly 1.0 at every logged step**, and `l_pers` stayed between 3.78 and 4.77
against a step-0 value of 4.175 — mean contraction 0.98×, i.e. none, which is
the only way a frozen codec can behave. Its scale-free forecast skill:

| steps | l_fore/l_pers |
|---|---|
| 120 | 0.727 |
| 600 | 0.634 |
| 1,680 | 0.595 |
| 5,000+ (n=50) | 0.479 ± 0.034 |
| last five points (~11k) | 0.436 · 0.447 · 0.436 · 0.432 · 0.430 |

**`--ref-fore` is therefore 0.44**, not the 0.58 read off step 1,680 while the
curve was still falling. The difference is not cosmetic. At 0.58 a treatment
that merely *matched* the control would log r_fore = 0.44/0.58 = **0.76** —
permanently below r_rec ≈ 1.0, so the smooth max would pick reconstruction at
every step and the run could not test its own hypothesis. At 0.44 it logs ≈
1.0 and the max is a genuine contest. #105/#106 were void for this reason
before they ever ran.

**Read that number carefully.** It is an in-sample MSE ratio in the codec's own
64-d z-space: `train_joint.py` samples only training windows (holdout years
2009/2017/2023 and the −45°/−25° longitude band are excluded from the pool),
so it is the objective's own value, not a generalisation estimate. In
typical-error terms it is √0.44 ≈ 0.66, a **34% smaller error** than
persistence, not 56%. And the model is fed month-of-year, so an unknown share
of it is seasonality — the same inflation that took RAPID r from 0.584 raw to
0.173 deseasonalised.

**Re-runs.** #107 (`lse@0.44`) and #108 (`sum@0.44`) with the live denominator
and the measured reference.

**Lesson, sharpened.** "Normalise by a quantity in the same space" is not
enough; the normaliser has to stay *differentiable*, or it is a constant
wearing a ratio's clothes. Reach for `.detach()` on a denominator only when
you have written down what the gradient does without it.

---

<a id="e-004-3"></a>
## E-004 · The normalisation was wrong — forecast never moved the codec

Chris asked a one-line question about the chart — *"how can reconstruction be
above 1?"* — and the answer turned out to invalidate the setup.

**The literal answer.** `r_rec` is current reconstruction ÷ the frozen codec's,
so above 1 means the codec now reconstructs *worse* than the one we started
from. Below 1 happens too (#94 hit 0.969) because the starting codec's LR had
annealed to ~0, so a fresh optimiser still finds headroom. The fixed-batch
probe is near-deterministic (1.000/0.999 in a smoke run), so the 0.97→1.11
swing is the codec genuinely moving, not measurement noise. #94 ended 3% worse.

**The real problem.** Reconstruction was normalised by the FROZEN CODEC (a
strong reference → sits at ~1.0); forecast was normalised by PERSISTENCE (a
weak reference → sits at ~0.3). The two were never on comparable footing, so
the smooth max was *always* reconstruction. Measured gradient weights on #94:

| r_rec | r_fore | weight on recon | weight on forecast |
|---|---|---|---|
| 1.033 | 0.257 | **0.957** | 0.043 |
| 0.969 | 0.414 | 0.902 | 0.098 |

**~95% of every step went to reconstruction.** E-004a was a reconstruction
fine-tune with a forecast garnish; it could not have tested the hypothesis even
in principle, whatever its probe number turns out to be.

**The fix.** Normalise BOTH against the same reference class — the frozen-codec
pipeline. Reconstruction ÷ frozen-codec reconstruction (already right), and
forecast ÷ *the same head's forecast on the frozen codec* (`--ref-fore`), not
persistence. Then 1.0 means "as good as the pipeline we started with" for both,
and "whichever is worse" finally means something.

That reference is not known in advance — it is exactly what the frozen-codec
control (#96) measures. **The control produces the number the treatment needs**,
so the sequencing is: control first, then the treatment normalised by it.

**Lesson, now a rule.** Chris's rule was "if one is high, the overall loss is
high". Applying it to two numbers that do not mean the same thing quietly turns
it into a single-objective run. Whenever losses are combined by comparison
rather than by weight, the references must be of the same KIND, and the
realised gradient split should be logged — a two-line check that would have
caught this before it cost two runs.

**The re-run cost three more dispatches before it ran a single step**, and none
of the three failures were about the science. Recording them here because the
log is supposed to explain why the calendar looks the way it does:

| run | what happened |
|---|---|
| #96 | `--require-resume` exits before the eval that writes `pixelmae.pt`, so the joint step had no codec to load. |
| #97 | Correct code, **wrong normalisation** (dispatched before `--ref-fore` existed) — cancelled deliberately at 45 min as the most expensive job that could not answer its own question. Its "still training, no curves" appearance was a separate defect: a joint run published its phase once, at the start of the *stage-1* step, and its metrics once, at the *end* of the fine-tune. |
| #98 | Landed on the box holding run-64/65/67 and was asked for run-62/63. The release-checkpoint seed added to cure exactly this **had never worked**: `"$TAG__pixelmae.pt"` expands `$TAG__pixelmae` (unset) so the pattern was `".pt"`, and it called `gh`, which is not installed on the Vast boxes — 127, stderr eaten by `2>/dev/null`, step green. |
| #99 | My dispatch JSON omitted `runner`, which defaulted to `ubuntu-latest`: a free 4-core CPU box, CPU torch wheel, no complaint from anything. Cancelled at 8 min. |
| #100 | Seeds `run-62.pt` from the release onto the wrong box in ~40 s and resumes. First resume-only Train step ever to succeed on a box that did not write the checkpoint. |

`runner` now defaults to `gpu`, so a down fleet queues visibly instead of
silently training on the wrong hardware; the joint step announces its own
phase and publishes curves every five minutes like the stage-1 step does.

---

<a id="e-005"></a>
## E-005 · Autoregressive unroll in the stage-2 loss (exposure bias) — SUGGESTIVE, ONE SPLIT

**Hypothesis.** Stage 2 trains on t+1 with TRUE context but is *evaluated*
autoregressively (`rollout.py`). A model that never sees its own errors during
training is not trained on the error distribution it faces at rollout, so
errors compound — the textbook exposure-bias failure. Rollout horizon is a
headline claim of this programme, currently measured on a model never trained
for it.

**Design.** `temporal.py --unroll U`. After the teacher-forced t+1 term, slide
the context forward on the model's OWN last prediction and add the next true
month's error, U−1 times, each weighted 1/(u+1) so a deep unroll cannot
outvote the anchor term. Gradient flows through the whole chain.

**Two things that had to be right, and were not at first.**
The window pool must guarantee the U extra months *exist* and are *train*
months — without that the unrolled steps would either index past the end of
the array or be scored on the holdout, and only the first of those would have
crashed. And the target after u self-fed steps is `Z[t+1+u]`, not the
teacher-forced target reused; verified by direct index arithmetic on a toy
tensor before any GPU time was spent.

**Control.** `--unroll 1` is bit-identical to the previous objective, and the
two statements above only look like they disagree. The extra-term loop runs
`for u in range(1, U)`, so at U=1 it never executes and the loss is exactly the
old `(pred − ztgt)²`; the `Z[t+1+u]` formula indexes *self-fed* steps, with
u = 0 being the teacher-forced term whose target is `Z[t+1]` — the original.
The pool guard collapses the same way: `t + U < T` becomes `t + 1 < T` and
`t_hold[t+1:t+U+1].any()` becomes `t_hold[t+1]`, which is what it always was.
So U=1 changes neither the loss, the weighting, nor which windows are eligible,
and the #88/#93 pairing below is a clean one-variable comparison.

**Result.** #88 (U=1, control, `head_sha` in the run record) and #93 (U=4).
Both resumed run-63 at step 60,000 with `--steps 60000`, so the stage-1 loop
ran zero steps and **the codec is byte-identical between them** — the embeddings,
the section, the holdout years and the 36 test months are all the same object.
Only the stage-2 objective differs. It is the cleanest paired comparison in
this log.

| | U=1 (control) | U=4 | |
|---|---|---|---|
| z MSE, model ÷ persistence | 0.494 | 0.641 | **worse** |
| channel MSE, model ÷ persistence | 0.576 | 0.682 | **worse** |
| RAPID r, deseasonalised | 0.173 | **0.449** | better |
| RAPID r, raw | 0.584 | 0.472 | worse |

**Reading.** Unrolling makes the model a *worse one-step predictor* and a
better *carrier of AMOC signal*. That is not a contradiction — it is what the
objective asks for. Optimising a four-step chain penalises error compounding,
which rewards capturing slow modes over fitting month-to-month variance, and
AMOC transport anomaly is a slow mode. The raw-r fall alongside the
deseasonalised rise says the same thing from the other side: raw r is inflated
by the seasonal cycle, which is exactly the fast, easy component an unrolled
model spends less capacity on.

**What this is not.** `rapid_r_deseas` comes from `temporal.py`'s **single
blocked split — n_test = 36 months**, not the year-blocked k-fold this log
argues from. At n = 36 the standard error on r is ≈ 0.13–0.16, so a single
0.449 is worth very little on its own. The pairing is what carries the result:
same codec, same features, same 36 months, one changed flag. Treat +0.28 as a
direction worth spending a k-fold on, **not** as a measured effect — and note
that the metric that moved is not the one the paper's headline uses.

> **DEAD, 2026-08-11 — E-010 measured the noise floor and this result is
> inside it.** Three seeds at U=1, everything else pinned, span 0.373–0.618:
> a range of **0.245**, against the +0.28 claimed here. Three seeds at U=4
> differ from U=1 by **+0.023 (t = 0.31)**. There is no unroll effect on the
> AMOC probe. What does replicate — here, in E-009 and in E-010 at t = 99 —
> is that unroll makes the FORECAST 29.7% worse. Retain that; discard the
> rest of this entry's conclusion.
>
> **Amended 2026-08-10, and the caution above was not cautious enough.**
> E-009 re-measured U=1 and U=2 under one code version on the year-blocked
> k-fold (~240 out-of-fold months), and **the two instruments order the arms
> oppositely**: k-fold U=1 0.555 vs U=2 0.377, single split U=1 0.413 vs U=2
> 0.465. Same runs, same features, same months. So the +0.28 reported here is
> not merely imprecise — the sign of the comparison is not stable under the
> resampling, and this entry's result should be treated as **withdrawn**
> pending E-009's remaining arms. What survives is the z-ratio direction:
> higher U forecasts worse, which E-009 reproduces.

**Next.** ~~Re-score both stage-2 models through `probe_kfold.py` rather than
the single split~~ — **struck 2026-08-10: that is not possible and asking for
it is what sent E-009 out with the wrong instrument.** `probe_kfold` scores
the frozen CODEC; #88 and #93 share one, so it returns the same number for
both by construction. The replacement is `rapid_probe_kfold` in
`temporal.json` — probe_kfold's year-blocked protocol applied to the HEAD's
own pooled hidden state, ~240 out-of-fold months with a block-bootstrap CI.
E-009 (#131–#134) re-measures U ∈ {1,2,4,8} on it; if the mechanism above is
right, the deseasonalised gain should be monotonic in U up to the point where
the 1/(u+1) weighting starves the anchor term.

---

<a id="e-004-4"></a>
## E-004 · BOTH joint runs RETRACTED — the collapse guard was measuring noise

**#86 (sum) and #91 (lse) are void.** Neither tells us anything about joint
training, and the confident conclusion previously written here for #86 — "the
predicted degenerate solution, observed" — is **withdrawn**.

**What went wrong.** The guard compared a *single training batch's*
reconstruction against a 20-batch mean, smoothed with an EMA. Per-batch
reconstruction is heavy tailed (random mask, 64 pixels), so the EMA tracks the
*mean* of that distribution, which rare spikes dominate. #91's own log is the
proof: every logged `r_rec` sat **below 1.0** (mean 0.88) while the EMA of all
steps sat at 1.02–1.07 and tripped at step 419. The guard was reading outliers
and it killed a healthy run 10% of the way in.

**Why the #86 conclusion is unsafe too.** Its `r_fore` falling to 0.036 is
*suggestive* of genuine degeneracy — that is a much larger move than #91's
0.33 — but it was stopped by the same broken instrument, so "collapsed" is
not a finding, it is an artifact of the stopping rule. Any real degeneracy
has to be re-measured.

**The fix — and then the removal.** Chris's response was the right one: *"i
never agreed to a trip wire… the loss should do the right thing."* Both parts
land. The tripwire was never in the design he asked for, and it is redundant
under it: the whole point of combining the terms so that "if one is high, the
overall loss is high" is that the objective self-corrects — the moment r_rec
rises above r_fore the smooth max **is** r_rec, and every following step pushes
reconstruction back down. A guard on top of that can only ever be a second,
worse opinion, and this one was demonstrably worse.

**No run in this programme is now failed by a metric threshold.** Reconstruction
on a fixed probe batch (fixed mask, eval mode) is still measured and logged
every `--check-every` steps, because reading what the codec did is the point —
it just stops nothing. If a run genuinely degenerates, that is a **result**: the
probe ladder scores the resulting codec and the number says so, which is far
better evidence than an abort ever was.

The measurement itself is now sound: the fixed probe
with a fixed mask, in eval mode**, every `--check-every` steps, and requires
**two consecutive** bad readings. Same batch, same mask: the only thing that
can move the number is the codec. Verified in a smoke run where the fixed
probe read 1.000 and 0.999 while the per-batch training loss swung 0.717–1.245.

**Two lessons, now rules.** (a) A stopping rule is an instrument and gets
validated like one — I validated the *loss* and the *index arithmetic* of these
experiments carefully, then shipped the statistic deciding whether a run lives
or dies without ever checking its variance. (b) Prefer *no* stopping rule.
Guards that kill runs on a threshold add a failure mode and remove evidence; a
run that finishes always tells you more than a run that was stopped.

---

<a id="e-004b"></a>
## E-004b · Joint training with the CONVENTIONAL sum loss — VOID (see above)

**Run** #86 (`joint`, sum, λ=1, warm-started from run-62's 40.7M codec, K=12).

Frozen references measured before any update: recon **0.2704**, persistence
forecast **4.1506**.

| step | r_rec (recon / frozen) | r_fore (forecast / persistence) |
|---|---|---|
| 400 | 1.639 | 0.108 |
| 800 | 1.139 | 0.074 |
| 1200 | 0.631 | 0.052 |
| 1600 | 0.932 | 0.042 |
| 2000 | 0.828 | 0.036 |
| **2153** | **EMA 1.112 → collapse, run stopped** | — |

**Conclusion — the predicted degenerate solution, observed.** The forecast term
falls to **3.6% of persistence**, which is not a forecasting triumph: it is the
codec making its own embeddings trivially predictable. Reconstruction pays for
it, drifting above the frozen value until the smoothed tripwire fired at step
2153. The weights are explicitly **not** a valid codec and were not probed.

**Why the sum loss does this.** Both terms are normalised, but they are not
equally *reducible*: r_fore can be driven toward zero by degrading the
representation, while r_rec cannot be improved much below 1.0 by any cheap
trick. A sum rewards whichever term moves most per unit of gradient, so it
walks straight into the cheap one.

**This is the case for the adaptive loss (E-004a), stated precisely.** With
r_rec ≈ 1 and r_fore ≈ 0.1, `max` and its smooth form select **r_rec** — the
objective that is *not* being satisfied — so the run cannot buy forecast skill
with reconstruction. Chris's formulation ("if one is high, the overall loss is
high") is exactly the property that forbids this collapse, and #86 is the
control that shows the conventional alternative needs forbidding.

---

<a id="e-004a"></a>
## E-004a · Joint stage-1+2 training (adaptive loss) — VOID, TO RE-RUN (see above)

**Hypothesis.** Stage 1 optimises *reconstruct this pixel-month*; nothing in
that objective asks the embedding to be **predictable forward in time**. If
that mismatch is what limits us, backpropagating stage 2's forecast loss into
the codec should improve field forecasting, and possibly the transport probes.

**Why now.** E-002 and E-003 closed both cheap scaling axes (below), so the
next move has to change the *objective*, not the size.

**Design.** Warm-start from a finished 40.7M codec, unfreeze it, minimise a
combination of reconstruction and forecast loss. Three loss modes:

| mode | combination | note |
|---|---|---|
| `sum` | `r_rec + λ·r_fore` | the conventional baseline, λ swept |
| `max` | `max(r_rec, r_fore)` | Chebyshev — whichever objective is worse, relative to its OWN baseline, is the one optimised |
| `lse` | `(1/a)·log(Σ exp(a·r_i))` | smooth `max`; default |

`r_i` are losses **normalised by their own reference** (`r_rec` = recon /
frozen-codec recon; `r_fore` = forecast / persistence), because a raw `max`
over unnormalised losses just always selects the larger-scale one and silently
becomes a single-objective run.

**Controls.** λ=0 (i.e. today's frozen-codec pipeline) must reproduce the E-003
numbers; forecast-only (λ→∞) is the deliberate collapse case.

**Tripwire.** Joint training can cheat by making embeddings *trivially*
predictable — the degenerate optimum is a constant embedding, forecast loss
zero, information zero. `r_rec` is logged every step and the run is called a
collapse if it degrades >10% from the frozen value, whatever the forecast loss
does.

**Scoring.** Probes stay **frozen-codec** on the resulting weights. If the
probe were allowed to fine-tune, the evaluation would no longer be a probe.

**Result.** _pending_

---

<a id="e-003"></a>
## E-003 · Does capacity help on the quarter-degree tensor? — NULL

**Run** #62 (`f3_anchor41M`, 40.7M params, 60k steps, 0.25° NA tensor, C=39).
**Control** #44 (`f3_pilot`, 0.92M, same tensor, same protocol).

| probe | r (k-fold) | 95% CI | RMSE |
|---|---|---|---|
| pooled ridge, **40.7M** | 0.631 | [0.513, 0.732] | 2.16 Sv |
| pooled ridge, **0.92M** (control) | 0.620 | [0.484, 0.741] | 2.25 Sv |
| unpooled attention head, 40.7M | **0.662** | [0.557, 0.745] | 2.10 Sv |
| head on **raw 3×3** (matched control) | 0.628 | [0.514, 0.729] | 2.17 Sv |
| wind-only bar, this tensor | 0.568 | [0.428, 0.696] | 2.29 Sv |

Other targets: MOVE 0.162 [−0.029, 0.340] (18-mo low-pass 0.516, wind-only
**−0.376**); Florida Current 0.012; OSNAP −0.060. Dip capture **51.2%**;
RAPID 18-mo low-pass 0.820; sign agreement 69.6%.

**Conclusion.** **44× the parameters bought +0.011** — inside seed noise.
Capacity is not the bottleneck on this tensor. The pretraining margin
(head 0.662 vs raw-3×3 0.628) is **+0.034 from one seed**; the 1° pair gave
+0.013. Same sign, twice the size, still a small fraction of its CI, and the
house rule forbids quoting a head number from a single seed.

---

<a id="e-002"></a>
## E-002 · Does training longer help? — NULL

**Run** #30 (`patch24_1M`, 1M steps) against the 40k/60k runs of the same
codec. Probe **flat from 50k to 1M steps**. Steps are not the bottleneck
either. Together with E-003 this closes both cheap scaling axes.

---

<a id="e-001"></a>
## E-001 · Does pretraining beat a supervised read-out on raw data? — NOT ESTABLISHED

Two seeds of `patch24` (1° tensor) probed with the unpooled attention head:
0.690 and 0.654, **two-seed mean 0.672**, against **0.659** for the same head
over raw 3×3 section tokens. Margin **+0.013** — indistinguishable from zero.
A directional "+0.031" reported from the first seed alone was **retracted**.

**Conclusion.** For the monthly RAPID probe, a supervised head over raw section
data recovers essentially everything the codec offers. The pretraining case
rests on the scoreboards where 240 labelled months have no answer: field
prediction, transfer to sparsely-labelled targets (MOVE), and calibration.

---

## Standing negative results (do not re-run without a reason)

- **Untrained codecs score near the wind bar.** A random-init encoder is a
  random-feature view of its own inputs, and the inputs include wind stress —
  so a step-0 probe lands at 0.33–0.57 depending on seed. Measured directly:
  ridge on wind channels alone 0.573, on a random *linear* projection of all
  channels 0.490 ± 0.03, on an untrained PixelMAE 0.334 ± 0.086 (5 seeds).
  The step-0 line is therefore **not** a zero-skill reference; it is
  approximately "these inputs, linearly, through a random lens".
- **The in-training light probe is a single split** (36 held-out months) and
  is far noisier than the k-fold. A decline in it across training is not
  evidence of representation loss — on #62 the light probe fell 0.546→0.380
  while the k-fold on the final weights read 0.631.
- **d_z sweep** settled at 64: 8/16/32 underfit, 128 was worse than 64.
- **MLP probe** trails the ridge on every target, for every codec — pointwise
  nonlinearity recovers nothing. The gain is from *unpooling*, not from
  nonlinearity.

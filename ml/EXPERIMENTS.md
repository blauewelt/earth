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
5. Numbers here are the **year-blocked k-fold** figures (`probe_kfold.py`) —
   the one instrument we argue from. In-training light-probe values are
   single-split and are noted as such wherever quoted.

Baselines, for reference. Wind-stress-only ridge in our own protocol:
**0.531** on the 1° tensors, **0.568** on the quarter-degree tensor. RAPID
monthly σ = 2.79 Sv, n = 240, ~68 effective DOF (~9 after an 18-month
low-pass).

---

<a id="e-007"></a>
## E-007 · How far past persistence can a FROZEN-codec forecaster go? — still improving at 24k

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
| **#110** | **24,000** | **0.406** | **0.473** | **+52.7%** | **0.319** |
| #112 | 60,000 | _running_ | | | |

\* single 36-month blocked split from the temporal hidden state, n_test = 36 —
noisy, and NOT the k-fold the programme argues from.

**Answer so far: no flattening.** Four times the steps bought **ten points** of
skill against persistence in real data units on held-out months, and the
deseasonalised RAPID correlation from the stage-2 hidden state nearly doubled.
6,000 steps — the budget every previous stage-2 run in this programme used —
was leaving a lot on the table.

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

**Consequence for E-006.** The bar the data-space loss has to clear is 0.473,
not 0.576 — and possibly lower once #112 lands. A redesign that merely matched
the old 6,000-step number would have looked like a win against the wrong
baseline.

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
## E-006 · The loss, rewritten in the data's units — DESIGNED, not yet run

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

**Status.** Designed, and deliberately not yet built. #109 (the twin) was
cancelled before it started rather than spend an hour on a superseded
formulation. Nothing else is dispatched against this until the algebra and a
synthetic smoke test are both done — which is the corrective for the actual
failure of 2026-08-09, namely building four times before thinking once.

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

**Next.** Re-score both stage-2 models through `probe_kfold.py` rather than the
single split, and add U=2 and U=8 — if the mechanism above is right, the
deseasonalised gain should be monotonic in U up to the point where the
1/(u+1) weighting starves the anchor term.

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

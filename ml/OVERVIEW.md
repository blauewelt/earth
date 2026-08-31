# The standing overview — every experiment, one line each, and what's next

**Last updated: 2026-08-31 ~08:55 UTC** (R0's second head DISPATCHED as #518 (E-062-R0b, the 7.6M arm) after the E-060a/E-060b heads were mirrored to the release — the "GCS blocker" is closed; the reboot handover from the chat session is checked in at `ml/handoffs/REBOOT_HANDOVER_2026-08-31.md`; previous stamp 2026-08-31 ~08:35 UTC: #516 COMPLETE — 879/879 in 21 h 26 m, `probes-516.json` archived, box stopped, and the 36-month rolls read; also #485/E-050 training-half harvest folded in — warm start survives, decoder audit still owed; previous stamp 2026-08-30 ~22:30 UTC: E-062-R0 landed: the first honest rolled number in this programme's history, and a metric correction that changes how every "corridor AUC" in this file reads; everything below the reset block is the earlier text, unchanged).

---

## THE PROTOCOL RESET — read this before anything below it

Chris, 2026-08-29: *"No need to compare models on the contaminated data.
Let's do a proper comparison. Also: I feel we need to reboot the programme as
until now we just evaluated how much for the training data the stage 2 can
learn by heart. It feels we need smaller models, and we need to re-evaluate
all dimensions as the previous eval results could not be trusted."*

The standing answer is
**[the protocol reset](https://blauewelt.github.io/earth/docs.html?f=ml/plans/PROTOCOL_RESET.md)**.
Four memorisation signatures, each from an artefact on `ml-metrics`: the pool
bug (21,018 supervised held-out targets, c25f6ff); **#510's band correlations
0.511 / 0.591 / 0.565 across 5–90 d / 95–180 d / 185–365 d — skill that does
not decay with lead is a recall curve**; **#513's corridor 0.838 on trained
longitudes against 0.176 on held-out ones** (gate head 0.804 against 0.058 —
on ocean it never saw, the head removes ~6 % of the anomaly variance, i.e.
essentially nothing, and every headline aggregate this programme has quoted is
a ~24 %-untrained blend); and **#513's dispersion — eight
independently-noised trajectories stay pinned together over years the model
has seen (0.1060 → 0.1015 across 36 months inside the record) and fan out over
years it has not (0.1974 → 0.4197 past its end)**, a memorisation test with no
labels in it.

**One correction to the framing:** smaller models are right for cost and not
for cause. E-060a's BEST held-out one-step loss is 0.60951 (7,597,856 params,
step 1,200) against E-059's 0.61049 (206,658,592, step 2,000) — 0.001 apart
across a 27× span, both peaking inside 2,000 steps and worsening for the next
198,000. (Restated 2026-08-30: the E-060 read-out falsified the stronger claim
that the arms match at a fixed later step — at 20,000 the small arm is 0.051
worse. Width buys how fast an arm gets there and how badly it then degrades,
not the level; and the degradation runs against the large arm, which collapses
RAPID 0.616 → 0.515 while 7.6M holds 0.598–0.611.) The constraint is 2,417
end-bins, as the E-061 plan says.

**Frozen protocol (§3), DECIDED by Chris 2026-08-30:** terminal holdout
**train ≤ 2020, test 2021–2024, NO GAP** (the gap was over-caution and is
withdrawn; 2025 does not exist — the tensor ends 2024-12-31) · `_trainlon`/`_holdlon` split as the headline,
never the blend · lead-decay as a falsifier · a null ladder on every number,
with a nearest-analogue retrieval baseline · early-stop at the held-out
minimum · rolled skill as the verdict, never a probe.

**Re-ranking programme (§4): R0 HAS LANDED — #516, 2026-08-30 21:42Z, and it
is the first honest rolled number this programme has ever had.** Verdict below.
R1 — re-rank cadence → stencil → unroll → znoise → FGN → width at the 7.6M
tier. R2 — E-061 keeps going; it is the only work aimed at the 43-year
constraint. **The 7.6M arm of R0 is UNBLOCKED and IN FLIGHT (08-31 08:49Z):**
`temporal_e060a.pt` and `temporal_e060b.pt` were mirrored from the TPU bucket
to `model-checkpoints-v1` as `head-weights-e060a-20k-window-s0.pt`
(30,425,836 B) and `head-weights-e060b-20k-window-s0.pt` (161,602,140 B), each
verified against its own `args` and md5-identical to the bucket object, and
**#RUNNUM2 (E-062-R0b, the 7.6M arm through #516's identical battery)** is rolling
on gpu-box-47898003 (#518, its first dispatch, died in 3 min on a full disk —
#516's stale 31.6 GiB scratch copy sat behind hygiene's 16 GB early exit;
fixed in `scripts/disk_hygiene.sh`). The GCS credential
was never a project blocker: `secret-handoff` round-trips in ~40 s.

### E-062-R0 (#516) · the verdict, 2026-08-30

**A METRIC CORRECTION FIRST, because it changes how every number above reads.**
What this programme calls "corridor AUC" is not an AUC. It is `horizon_auc` =
mean **`msss_clim` = 1 − MSE_model / MSE_climatology**, and the fields are
already anomalies, so "climatology" = predicting zero anomaly. **1.0 perfect,
0.0 exactly as good as predicting nothing, negative = error exceeds the anomaly
variance.** Negative is NOT "below chance" and does not mean the ranking
inverts. `ml/rollout_spatial.py:117` has said so since E-017.

**The headline: the clean-pool head PASSES the lead-decay falsifier its
contaminated twin FAILS.** Field anomaly correlation `acc`, corridor scope,
same battery, one variable (the training pool):

| | h=1 (5 d) | h=73 (365 d) | mean |
|---|---|---|---|
| #516 · E-059, clean pool | **0.606** | **−0.031** | 0.105 |
| #510 · E-051, contaminated | **0.985** | **0.973** | 0.946 |

A near-unity field correlation twelve months out, flat across the year, is a
memorised trajectory being replayed. The clean head decays like a forecast.
Corridor `horizon_auc` goes +0.888 → **−0.439**; gate −0.395; window −0.401;
unpooled bands 0.107 / −0.242 / 0.163 (all inside noise of zero at ~9 effective
starts — read them as "no measurable transport skill", never as inversion).

**The negative score is a CALIBRATION failure and the arithmetic closes.** Mean
`acc` 0.105 against mean `amp_ratio` 0.780 — the head keeps emitting anomalies
at 78 % amplitude long after it stops knowing their sign. The identity
`msss = 1 − (1 + a² − 2a·ACC)` reproduces all 73 corridor leads to a mean
absolute error of **0.0135**. Amplitude-calibrated (`a = ACC`), the same rolled
states would score **+0.019** instead of −0.439. The head also **beats
persistence at every lead but the last** (mean msss_pers +0.204).

**Where the residual skill is:** SST is the only channel beating climatology
past one step, out to 90 days (+0.069); SSH is best at 5 d (+0.717). And **only
8 of 40 channels are scored at all** — the 32 `rg_*` Argo channels are null at
every lead, the evaluator's own confirmation of the data ladder's §2 finding
that they are 80 % of the tensor's bytes carrying ~0.28 GB.

**Caveats that must travel with these numbers:** n = 1 at a cadence with no
replicate pair (§3b) · the head is the 200k memorised end state, past its
held-out minimum at step 2,000, so this is a floor (§3.5) · it still scores
against the OLD interspersed holdout · E-059's own probe was 0.522, below both
wind-stress ridge bars · the pentad r2 tensor has no longitude hole, so
`_trainlon` equals the parent and the spatial split was not measured here.

**THE RUN IS COMPLETE (08-31): 879/879 in 21 h 26 m, `probes-516.json`
archived at 2,041,999 B with NO `in_progress` marker, and every number above —
published from the §5.25 partial — is BIT-IDENTICAL in the final artefact.**
The box is stopped and verified `exited`.

**The 36-month rolls added one more thing.** They are NOT dispersion blocks —
E-059 is deterministic, so one trajectory and no spread, which means §2d's
memorisation-by-dispersion test is simply UNANSWERED for a clean-pool head
rather than answered negatively. What they do carry: a 3-year free run from
2004-12 correlates **0.636** with the true transport at pentad resolution but
**−0.13 on an 18-month low-pass**, at 61 % amplitude — so the skill lives at
high frequency and the multi-year band, which is the AMOC question, is
unmeasured by this roll (two effective cycles; −0.13 is not a level, and
2004–2007 are training years anyway). That is a new, data-driven argument for
the terminal holdout: four contiguous held-out years is the first window that
could measure it. Separately the 3-year roll PAST the record stays bounded
(sd 1.248 against 2.208 in-record, no drift to a rail) — non-divergence is a
precondition for forecasting forward from now, and it is not automatic.

**Next, in order:** (1) roll the step-2,000 checkpoint — cheapest untried thing
in the programme; (2) fix the amplitude calibration, a decoding change worth
~+0.46 msss on states we already have; (3) drop or thin the 32 upsampled Argo
channels; (4) the terminal-holdout retrains at 7.6M — unchanged, no longer
needing to rescue anything, and now the only route to a low-frequency number.

Full reading:
[E-062 in the experiment log](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-062).

**Retired (§5):** every rolled number from a pre-c25f6ff head as evidence of
forecast skill · E-057's F1 · the capacity ladder as a skill question ·
E-057 seed 1.

---

**E-054b is now running the
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
bar. **THE HOLDOUT BUG IS FOUND AND FIXED, AND THE CONTROLLED RETRAIN IS
RUNNING.** Chris's question ("where does measured 2009 leak in?") exposed
that every archived stage-2 head was teacher-forced on the held-out years'
transitions: the loss is dense over the window while the pool only checked
the FINAL target (`win_ztgt` temporal.py:2819 vs `ok_t` :2889; count proof
2,779 = 3,142−219−144 exactly). Fix: `--holdout-scope window` (c25f6ff) —
pool keeps only windows touching NO holdout bin, self-certifies by brute
force before training, and the legacy pool survives bit-identically as
`endpoint_contaminated` so the archive is reproducible. **E-059** (E-051's exact twin, one change: the pool) is
training on us-west4-a spot since 17:09Z and its **first-minutes checks
PASSED to the digit** — 2,417 end-bins, 209,549,066 windows, certificate 0
violations, val_persistence 21.44621, all registered in the plan beforehand;
the one-step gap vs 0.0330/0.0298 IS the h=1 memorization term. **FIRST READ
AT 24k/200k: the train curves are twins and the val curves are not.** Same
architecture, seed, codec, Z, validation windows (val_persistence 21.44621 in
both, six digits) and hardware — only the pool differs — and E-059's train
z-mse tracks E-051's (1.2352 vs 1.2126 at 24k, so the 13% supervision cut
costs nothing visible) while E-051's val ratio falls monotonically
0.2247→0.0748 and **E-059's does not fall at all**: 0.6105 at 2k, 0.6410 at
24k, drifting up. A first read, not a verdict — 176k steps and the whole LR
decay remain, and the RAPID probe is meanwhile indistinguishable (0.616 vs
0.612 at 20k), which is the counter-evidence. Pace corrected from the
estimated 12.6 h to E-051's measured 16.45 h for 200k, so **phase 2 is owed
~10:25Z 08-29**, wake scheduled; the ~48-min post-probe pause is in E-051's
record too and is normal.
**The E-059 plan file was committed EMPTY twice** (fc83585, 2d1b20a) — the
write never landed and neither commit noticed, because docs.html registers
the path, not its size; reconstructed and pushed in 53cb723. **Scopes
amended (58eb286, Chris): the leaky pool is renamed
`endpoint_contaminated`, the DEFAULT is now `window`, and a third scope
`target` masks held-out targets while still admitting held-out context.
Measured cost: contaminated scores 400,176 frame-targets of which 21,018 are
held out; `target` 379,158 (−5.25%, 0 held out); `window` 348,048 (−13.03%,
0 held out) — strictness costs 7.8 points, not a factor.** **#508** re-rolls the OLD 398k
head with the battery shortened to 36 months each way (879 steps, ~21.3 h,
inside the token) — its future roll past the record's end is the direct
recall falsifier — dispatched as **#510** after #508 and #509 died on the
same full disk two layers deep; verified on the roll at t0 19:18:29Z with the
protocol matching #503 field for field and `longm:36,futm:36` correctly
converted to 219+219 axis steps. Its seeding overran by an hour, leaving only
~50 min of token margin on the future battery, while the 441-step determinism
certificate lands ~05:58Z, safely inside. #503 was cancelled 16:24Z (its 40 h timeout could never
deliver the 20-year battery); its scored partial is fully harvested. The
monthly 0.939 champion carries the same question until its own
scope=window retrain.

**E-056 IS RESOLVED (one-step): the token road does NOT open.** #507, the
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
z-scored anomaly field has no physical story. Mechanism CORRECTED ~15Z (Chris's question exposed the first one as wrong —
the roll breaks at the year boundary, `rollout_spatial.py:880`, so no scored
target is a training bin): the leak is in TRAINING — the stage-2 loss is
dense over all K=144 window positions (`win_ztgt`, temporal.py:2819) while
the pool excludes only windows whose FINAL target is held out (:2889; count
2,779 = 3,142−219−144 exactly), so windows straddling a holdout year
teacher-force that year's measured transitions into the weights, of order
tens of times per pixel. The held-out year is held out as an ENDPOINT, not
an experience — and the monthly champion (K=24 months, same 2-year span,
same pool) has the identical structure. Report it as a
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
| **#RUNNUM2** E-062-R0b · the 7.6M head `head-weights-e060a-20k-window-s0` through #516's identical battery | does the small-tier clean-pool head forecast as well as the 206.66M one on the rolled field — was capacity ever the axis? | shape: lead-decay must PASS; level: #516's acc 0.606@5d → −0.031@365d, corridor msss_clim −0.439, amp_ratio 0.780, msss_pers +0.204, SST +0.069 to 90 d. Parity or better ⇒ small tier is default for cause | gpu-box-47898003 (Vast 49242934), started DISPATCHTIME 08-31 (after #518's 3-minute disk death at 08:52Z); token deadline DISPATCHTIME+24 h; expected well under #516's 21 h 26 m |
| **#504** E-056a-R token substrate K=24 (+ E-056a-CLEAN twin registered) | is the E-050 warm-FSQ 16-bit-per-pixel-bin alphabet a competitive forecasting substrate? Overnight both arms died: #494 CANCELLED at step 2800/20000 by an actor outside this session; #495 VOID — CUDA OOM on its first forward (K=144 batch 256 does not fit a 24 GB 4090; its control #478 ran that batch on a bigger card, so E-056b needs an 80 GB box, NOT a halved batch). Codec now durable on the release; token-Z cache durable ⇒ embed is free | ≲0.44 ⇒ token road opens at ~5% state size; ≳0.50 ⇒ quantization lost the forecastable signal. Denominator is the TOKEN-scale 0.63451. znoise-dose confound pre-registered (0.879 vs 0.151 rel_pers) — the CLEAN twin at znoise 0.12 settles it in the same wave | **TRAINING CURVE DONE 09:42Z: final ratio 0.53873** (0.34183/0.63451) after 0.600@2800 / 0.574@6000 / 0.548@16000 — flat from ~16k, WORSE than the continuous control 0.5056 and far from the 0.4394 lattice bar. Not a verdict: ~5.8x the intended relative dose. Probe ladder tail still running |
| **#505 re-dispatch** E-056a-CLEAN, K=24, znoise **0.12** | is #504's 0.555 the SUBSTRATE, or the 5.8x noise handicap? Identical to #504 in every other field | the dose-matched level against lattice 0.4394 / continuous 0.5056, denominator 0.63451. If CLEAN materially beats 0.7 the dose was the handicap; if they agree the substrate verdict stands at either dose | **RE-DISPATCHED as #507 at 10:19Z, QUEUED on gpu-box-32966687 behind #504** — deliberately the SAME box as its 0.7-dose twin, so the pair differs in the dose and nothing else (three other hosts were tried and failed: one full disk, two `resources_unavailable`, one that never registered its runner in 18 min). **RUNNING since 11:24Z, healthy at 0.306 s/step, gpu 99.99%** — ratio 0.590 at step 2,800 against #504's 0.600 at the same step, so the dose looks like a small effect so far; done ~13:10Z plus its ladder |
| **#506** E-056b token substrate at K=144 — **CANCELLED 12:30Z, HELD** | does the dense two-year slab hold up on tokens, and what does it cost? | vs #478's 0.0820 — unanswered | the rented H100 never used its GPU: ten `gpu_util` samples of 0% at `cpu_util` 95-97% across 3 h 55 m (the dead-frame check never fired, so they were real), and zero step records in 50 min where #478 at identical geometry writes its first at `wall_s 240`. Cancelled on a rule written 25 min ahead; ~$7.95. **Re-dispatch only if #507 rehabilitates the substrate** |
| **#500 → #511/#512** E-057.1a FGN seed 0 + its harvest chain (#502 seed 1 DIED at ~163k/200k — its box's host dropped offline 19:20Z; head unreachable on that disk; re-run deferred to seed 0's verdict per §3b winner-replicate logic, e-057(k)) | does a LEARNED perturbation + fair CRPS (eps^32 conditional LN, N=2, znoise OFF) replace the hand-dosed znoise and un-damp the roll? | **F1 lands with #512, the FIRST M=8 ensemble roll** (queued behind #500 with #511 headpub): ensemble-mean corridor AUC vs znoise pair 0.7235 / clean 0.6781 — same contaminated pool on all three sides so the CONTRAST is clean under the E-059 caveat; F2 telemetry stayed healthy to the end (spread_ratio plateau ~0.5–0.75); F3 = the 36+36-month dispersion battery in #512 | #500 trains blind past its expired token, done ~23:15Z; #511 publishes the head + rescues metrics; #512 ≈34–40 h (M=8 = 8× roll cost) ⇒ F1 ~mid-day 08-30; §5.25 partials on ml-live-512; hand-harvest at the end |
| **E-054b** ~400M capacity rung (1280×20, K 144, 200k fresh, TPU spot, grad-accum 4) | does capacity, not steps, buy the next factor at full pentad span? | vs E-051's 0.0330 at 200k/206.6M (the step-matched control — E-054a's 0.02981 is a 400k number); first-minutes verdict **PASSED on knobs** (`K 144 · lr 4e-4 · 1280x20 · stencil 145 · znoise 0.7 · grad_clip 128 · grad_accum 4 (micro 64)`, Z pulled and VERIFIED 16.24 GiB); **and the first training step at 08:37Z showed NO OOM** — params_M 399.948, grad_accum 4, micro_batch 64, and val_persistence 21.44621 bit-identical to E-054a's denominator, so the ratio is directly comparable to E-051's 0.0330 | first launch OOMed 00:15Z (registered risk fired) ⇒ grad-accum built + certified exact (2.4e-07) and pushed; a 07:08Z relaunch ran the WRONG config for ~1 h and was killed; **relaunched 08:10:33Z us-west1-c spot, correct** — ≈32 h ⇒ ~16Z 08-29 |
| **#503** E-051 roll (398k K=144 head, day-matched, replay battery, FIRST roll with E-055's unpooled keys) | does the best pentad one-step ever (0.0298) survive a 12-month roll, where the small pentad head collapsed to −0.499? | vs monthly _trainlon 0.939/0.939 and pentad-K24 −0.499; the battery (tracking-vs-lead profile; flat = calendar replay) is MANDATORY before the number may be called forecast | measured 06:50Z: 87.1 s/step. **Decisive corridor AUC at step 657/3363 ≈ 16:50Z 08-28**, inside the 23:52Z token; the 2,922-step battery needs ~71 h beyond it — ride-vs-cut is a decision for Chris |
| **#485** E-050 warm-start FSQ | does a trained encoder survive quantization where every cold start collapsed? | decoder-ceiling audit (Falsifier B): fast channels inside the 9–19% FVU band on Argo-free bins | **HARVESTED 08-31 (training half): Falsifier A did NOT fire** — fits stayed input-dependent through 260k (std_med 0.97 → 0.85 → 0.80, rms 1.0), loss_rec 0.270 vs parent #480's 0.229 (quantization tax ~18%), chan_mse 0.742 vs #480 0.681 vs persistence 0.843, head probe 0.588 vs #480 0.579 (consistent, n=1). Codec durable as run-485__pixelmae.pt. **Falsifier B (decoder-ceiling FVU) still NOT run** — the only open E-050 read-out |
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
| E-050 | warm-start quantization | WARM START WORKS as training: #485 reached 260k with no collapse signature (the cold-start disease is a cold-start disease) at an ~18% loss_rec tax over the continuous parent; decoder-ceiling audit (Falsifier B) still owed — see §1 |

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

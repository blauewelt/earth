# earth / ml — standing instructions for the training & research work

This file governs everything under `ml/`, `scripts/` that touches the fleet,
and `.github/workflows/ml-train.yml`. The repository root `CLAUDE.md` governs
the globe app (layers, tiles, UI, deploys) and its rules — "deploy first",
"stamp before you commit", the layer checklist — **do not apply here.** They
were written for a static site where a bad deploy is reversible in ninety
seconds. A bad dispatch here costs hours of rented GPU and, worse, can produce
a number that looks like a result.

Chris asked for the split on 2026-08-10, after a day in which the two kinds of
work interleaved badly.

Read alongside:

| file | for |
|---|---|
| `docs/ML_BASICS.md` | what the system IS and what its numbers mean |
| `docs/INFRASTRUCTURE.md` | the fleet, its failure taxonomy, its invariants |
| `ml/EXPERIMENTS.md` | what was run and what it returned |

---

## 0 · The three rules that would have saved the most time

Everything below elaborates these.

1. **Verify the ARTEFACT, not the intention.** #119 spent 93 minutes embedding
   and then refused, because the checkpoint it was told to continue contained
   no optimiser state. The design had been written from what the file was
   *assumed* to hold. Opening it would have taken ten seconds.
2. **A step that reports success is not evidence it did anything.** Assert the
   effect. Four separate steps in one night reported success while doing
   nothing.
3. **Check a precondition where the inputs are all it has cost you.** A guard
   at the point of USE is correct and expensive; the same guard at dispatch is
   correct and free.

---

## 0b · Model economics: the session plans, subagents implement

Standing rule, Chris 2026-08-18: *"make sure that it is you who plans
(Fable), but that you use subagents (with Opus 5) for the implementation
details"* — after one long session burned ~10% of the weekly Fable quota.

The split: the MAIN session does the thinking that benefits from the larger
model — experiment design, hypothesis and falsifier statements, diagnosis of
surprising failures, dispatch decisions, reading results, anything touching
EXPERIMENTS.md's claims. SUBAGENTS (Agent tool, `model: "opus"`) do the
implementation details — writing and editing code and tests to a stated
spec, pulling and summarising logs, monitoring runs, mechanical refactors,
doc formatting. Subagents share the working directory and credentials
(`/home/claude/.gh_pat` etc. are on disk), so monitoring scripts run
unchanged.

Two cautions from the day the rule was made. A subagent's report is an
intention until verified — the main session still runs the tests and reads
the diff before committing (§0.2 applies to subagents exactly as it applies
to workflow steps). And diagnosis stays in the main session: the #366
green-but-void run was found by noticing a 2.6 h "success" that should have
taken 11 h — the kind of surprise a summary can smooth over.

## 0c · A run number never travels alone

Standing rule, Chris 2026-08-19: *"For these experiment pages, can you make
sure they contain the summary from the status page? (as a first step, just
make it a standing rule to provide a short summary next to each experiment
number) And then maybe links could go to the status page or the
experiments.md (which have summary and curves)"*

He was reading a session report on a phone. It named **#413**, **#411** and
**#410**, and linked each one to its GitHub Actions page — the single surface
that can tell a reader neither what the run was FOR nor how it is doing. The
summary had existed the whole time: every dispatch carries a `doc` string, and
the status page renders it under each run (`docOf()` / `expTag()`). The report
threw that away and posted the key instead.

**(a) A run number never appears alone.** Every `#NNN` written anywhere — a
chat reply, `ml/EXPERIMENTS.md`, a `claude/*` hand-off, a commit message, the
dispatch `doc` string itself — carries a short summary in the same breath. The
canonical form is **`#413 (E-035 seed-0 roll-forward)`**: the run number, then
the experiment ID where there is one, then a few words naming what the run
DOES. Not its hypothesis, not its result — its job.

- ✅ `#410 (E-038c daily codec, re-run)`
- ✅ `#411 (E-042 SST arm on the r2 tensor)`
- ✅ `#409 (E-038a's own codec through the full probe ladder)`
- ❌ `#413` — an opaque key
- ❌ `#413 finished` — says nothing a reader can act on; every run finishes

A run number is a database key with no meaning outside the database. The reader
cannot resolve it: not on a phone, not mid-conversation, and not later — by the
time a number reaches a hand-off doc the session that knew what it meant is
gone, and the only surviving copy of that knowledge is the clause you did not
write.

**(b) Link the reader, not the raw CI page.** Root `CLAUDE.md` §0b already
sorts link targets by what the reader needs to SEE — blob URLs for source,
Pages URLs for rendered things, `docs.html?f=` for markdown. Runs get their own
tier, and it exists because **the status page and `ml/EXPERIMENTS.md` are the
two places that carry a summary AND the curves.** Every other target carries
one or neither.

- An **IN-FLIGHT or recent** run → the status page's per-run deep link:
  `[#411 (E-042 SST arm)](https://blauewelt.github.io/earth/status.html#run-411)`.
  That anchor resolves only while the run is still inside the page's fetch
  window; once it has aged out the link lands on the page but not on the run,
  which is why a settled run belongs in the next tier and not this one.
- A **SETTLED** experiment → its section in the log:
  `[E-042](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-042)`.
  Use the explicit `<a id="e-042">` anchor, never a heading-derived one — these
  headings change as results land ("DISPATCHED" → "RESOLVED"), so an auto anchor
  rots on the first edit.
- A raw `https://github.com/blauewelt/earth/actions/runs/<id>` URL **only when
  the point genuinely IS the CI log** — a stack trace, a step timing, the
  `INPUTS_JSON` block. **This is the exception, not the default.** The Actions
  page needs an authenticated desktop browser before it is worth opening at
  all, and even then it shows a reader nothing about what the run was for.

The two halves are one rule: the summary says what the number means, the link
says where to go and watch it.

## 0d · An experiment description is a config, not a story

Standing rule, Chris 2026-08-19: *"Make it a standing rule to have all
experiments descriptions be: a) Structured, with fields: Num params, stage:
encoder, data, ... (you could even automatically render an experiment config).
b) Absolute, not relative: Not 'restart of #101' but 'Add SST to training data
(restart of #101). Params: ...'"*

§0c fixed the run number that arrives without a summary. This fixes the
summary that arrives without a configuration.

The incident is #409. Its dispatch `doc` string read **"RE-DISPATCH of #407"**.
#407 was itself a re-dispatch — of #406's read-out protocol against #386's
checkpoint — so a reader who wanted to know what the run TRAINED had three hops
to make: #409 → #407 → #406/#386, two of them into descriptions that were also
relative, and one into a cancelled job whose log blobs had already expired. The
answer, after three hops, was that #409 trained nothing at all: it was
eval-only, a probe ladder over a finished checkpoint. That fact sat in
`provenance.json` the entire time, and in no sentence anybody would read.

**(a) Every dispatch `doc` string and every EXPERIMENTS.md dispatch entry opens
with a STRUCTURED header line, before the prose.** Fixed order, `·`-separated:

> experiment ID · what the run does, in ABSOLUTE terms · `params` · `stage` ·
> `data` · `arch` · `steps×batch` · `resume`

- **`params`** — parameter count, e.g. `38.0M`.
- **`stage`** — one of `encoder`, `stage-2`, `eval-only`, `headpub`, `sroll`.
  This is the field that would have answered #409 in one word.
- **`data`** — the tensor BY NAME, e.g. `family4_na025_pentad`, never "the
  pentad one".
- **`arch`** — the geometry, e.g. `512×12 d_z 32 patch 1`.
- **`steps×batch`** — and where nothing trains, say so in the same breath
  rather than printing a number that looks like a budget.
- **`resume`** — what it seeds from, BY NAME (`run-386`, `f3_anchor41M`), never
  a bare `!` reference the reader has to expand.

❌ **"RE-DISPATCH of #407"** — relative, and the reader must chase #407, which
was itself a re-dispatch of something else. Three hops to learn what trains.

✅ **"#386's pentad codec through the full probe ladder + unpooled head
(re-dispatch of #407) · params 38.0M · stage eval-only · data
family4_na025_pentad · arch 512×12 d_z 32 patch 1 · steps 166752 (= checkpoint,
nothing trains) · resume run-386"**

**(b) The relative clause is allowed — in parentheses, AFTER the absolute
description.** Genealogy is useful once the reader already knows what the run
does; it is never the description itself. Chris's own shape: *"Add SST to
training data (restart of #101). Params: ..."* — the change first, the ancestry
second, the config after both.

**(c) None of this is new data collection — it is placement.** `provenance.json`
and `plan-*.json` already carry params, stage, tensor, geometry, step budget and
resume for every run in the fleet; §1's recipe mechanism already refuses a
dispatch whose architecture contradicts its checkpoint. The rule is about
putting those fields where a reader's eyes actually land: the one line that
appears in chat, on the status page, and in a hand-off six sessions later. The
machine half of the same rule is that **the status page renders a config line
directly from provenance under each run**, so the fields that can be generated
are generated, and the fields a human must choose — what the run DOES, in
absolute terms — are the ones the `doc` string is for.

## What this programme is building

Standing statement of purpose, Chris 2026-08-19: *"What we're building is a
predictor for everything, so it doesn't matter whether the embedding will
contain more or less information than the raw pixels. The embedding makes large
chunks of data 'attendable' by a transformer. And we can predict everything
from predicting embeddings (not just AMOC). That's the overall plan."*

Read this before designing an experiment. It decides which comparisons are
worth GPU and which are answering a question the programme is not asking.

**1 · The object is a predictor of the WHOLE STATE, not an AMOC probe.** What
is being learned is a forward model of the North Atlantic state — every channel,
every pixel, rolled forward in time. AMOC transport at 26.5°N is the HEADLINE
metric because RAPID is the best-instrumented read-out available to us: a long,
continuous, physically meaningful series against which a forecast can be scored
at all. That is a property of the INSTRUMENT, not of the target. AMOC is one
read-out of a general forecast, and a design that improves the AMOC probe while
degrading the forecast has moved the programme backwards.

**2 · The embedding is not required to carry MORE information than the pixels.**
It cannot. It is a compression; a compression's information content is bounded
by its input, and any experiment framed as "does the embedding beat the pixels
on information content" has a known answer and does not need running. The
embedding's job is **ATTENDABILITY**. The 84,405 active ocean pixels × 39
channels of one quarter-degree time step are not a sequence a transformer can
attend over; a token sequence is. The daily family-5 tensor is **165.6 GB** of
raw pixels and cannot be attended at all — not at any batch size, not on any
box we rent. The
codec is what turns the state into a sequence, and that is the whole of its
mandate: make large chunks of data attendable, cheaply enough that the
forecaster can roll them forward for twelve months.

**3 · Therefore embedding quality is judged by what stage 2 can predict FROM
it.** The operative instruments are rollout skill, corridor AUC and band
correlations — exactly how E-022, E-035, E-036 and E-037 already score. Those
are the numbers that say whether a representation is a good SUBSTRATE FOR
FORECASTING. A current-state probe-vs-raw delta is not that number, and must
not be used as one: it asks what today's embedding says about today's transport,
which the forecaster never has to do.
Those instruments are not equally reproducible, and §3b prices each of them: on
the same xl checkpoints the corridor reproduces across a seed pair to
0.002–0.005 while the transport band correlations spread 0.05–0.07. Choose the
instrument the claim can survive, and read §3b before deciding how many seeds
the arm buys.

**4 · A raw-pixel head is a legitimate READ-OUT control, not a rival
architecture.** E-038a used it exactly right — the raw-3×3 control is what
exposed the pooling artefact and what kept "the codec knows this" separable from
"any read-out with spatial structure knows this" (`docs/ML_BASICS.md` §5). Keep
it. But when it comes back at parity, the finding is about READ-OUT DESIGN. It
is not evidence that the codec should be replaced by raw pixels, because there
is no raw-pixel forecaster to replace it WITH: the question a raw head answers
is "what does today's state say about today's transport", and the programme's
question is "what does the state do next".

**5 · Predicting embeddings predicts everything at once.** Any quantity readable
from a real embedding is readable from a predicted one — transport at 26.5°N,
Florida Current, MOVE, OSNAP, SAMBA, and equally any field we have not thought
to probe yet, because the decoder is already there. That is the plan, in three
steps: **encode → predict forward in embedding space → read out anything.** The
read-out is the cheap, replaceable end of the system. The forward model in
embedding space is the programme.

---

## 0e · Never park the fleet over budget — report it and keep going

Standing rule, Chris 2026-08-19: *"please don't worry about top ups and proceed
in spite of remaining budget, I will make sure the top ups are happening in
time."* Earlier the same evening: *"I really want to make progress without any
holdups... do not hold up the fleet over night, just count on my top-up."*

So the budget is HIS constraint, not a session's. A session must never decline
a dispatch, cancel a healthy run, or stop a working box because credit looks
short. What it owes him instead is **arithmetic, early**: the balance, the
burn, the implied runway, and which specific runs finish inside it and which do
not. That is actionable; "I paused the wave to be safe" is not — it costs a
night and produces nothing.

This does NOT relax the two rules it sits beside, because neither is about
saving money:

- **Idle burn is still stopped on sight** (§7, and the hourly fleet-health
  check). A box running with no job is not progress at any price. Note the
  ordering trap this created on 2026-08-19: five boxes were started for the
  no-holdout wave, the hourly check fired in the gap before they had jobs, and
  correctly stopped all five as idle burn. **Dispatch first, then let the watch
  resume** — or disable it for the provisioning window and re-enable it the
  moment the jobs are live.
- **A run that cannot answer its question is still cancelled** (§4.13, §11).
  Killing #410 and #411 mid-flight when the holdout regime changed was right,
  and it was a scientific judgement, not a financial one. Preserve what the
  dead run measured before you kill it — #411's SST evidence survives in
  `ml/recipes/f4r2-40M-nolonhold.json`'s `_provenance` precisely because that
  was done before its branch was deleted.

The failure this rule forbids is the quiet one: a session that sees a short
runway, silently scales the wave down to fit, and reports a tidy plan whose
smallness nobody ever questions.

## 1 · Before you dispatch

- **State what result would FALSIFY the hypothesis**, and check the
  configuration can produce it. Several runs were healthy and could not, by
  construction, test their own hypothesis.
- **Write the EXPERIMENTS.md entry at dispatch**, hypothesis first, so the log
  cannot be rewritten to fit the answer. Name the control explicitly.
- **Open every checkpoint the dispatch names** (`scripts/precheck_stage2_head.py`
  does this for stage-2 heads) and confirm it supports the mode you are asking
  for.
- **Size the job against its own timeout.** Measured: 0.419 s/step for a 192×4
  head at K=24. So 24,000 steps fits in 350 minutes and 60,000 does not, and
  200,000 needs ~23.3 h plus the embedding. `job_timeout` is an INPUT, not a
  cap — self-hosted runners can run for days.
- **Publish the plan** (`scripts/publish_plan.mjs <n> '<json>'`) so the status
  page can draw the schedule before the run spends anything.
- **`runner` defaults to `gpu`.** Omitting it used to send jobs to a free
  4-core CPU box where a 40M codec "trains" and nothing says wrong-hardware.
  Check `runner_name` on any dispatch you care about.
- **Name a recipe; never hand-assemble an architecture.** On #358 — an
  entirely ordinary run — **17 of 24 inputs had to be overridden**, so every
  dispatch was a 24-field copying exercise, and copying exercises are what
  produced #395 (resume without a width: sixty `size mismatch` lines in 90 s)
  and #387 (`codec_heads` left at 4 while `d_model` went to 1024: head_dim
  256, and the 202M codec's embedding collapsed at ~step 12k while `loss_rec`
  hid it). A default that is never correct is not a default. Three mechanisms
  replace the vigilance, and they are load-bearing rather than advisory:
  - **`window: recipe:<name>`** expands a checked-in `ml/recipes/<name>.json`
    into the full parameter set. Recipes carry `_description` and
    `_provenance` — which run measured this configuration — and a key the
    workflow does not actually read as `$RECIPE_<KEY>` is REFUSED, so a
    setting cannot appear to apply and quietly do nothing.
  - **`--resume` DERIVES the architecture** from the checkpoint's own `args`.
    Do not restate a width the file already holds; a dispatch that
    contradicts its checkpoint now refuses before the model is built.
  - **Nothing falls back to the pilot.** An unset architecture with no recipe
    and no resume refuses in seconds, naming the flags it wants.
  `python3 tests/test_train_config_guards.py` pins all of it, and
  `python3 tests/test_workflow_config.py` pins the workflow side — including
  the 21,000-char ceiling that took every dispatch in the repo down for ten
  minutes on 2026-08-17.
- **A collapsed codec is invisible to every monitor we had.** `loss_rec`
  cannot see it (the decoder rescales freely, so reconstruction stays
  mediocre-but-finite while the latent runs away — #387 sat at 0.27–0.32 for
  9 hours after it died) and the fleet health checks cannot see it (a dead
  model holds the GPU at 100%). The probe correlation can, because a
  correlation is scale-invariant: `--collapse-r` (default 0.05) aborts after
  two consecutive sub-threshold probes. A NaN probe is NO READING, not
  collapse — instrumentation must never be the thing that loses a job — and a
  genuinely non-finite loss is caught separately in the training loop.
- **A dispatch input you omit is not "inherited" — it is the DEFAULT.** The
  workflow's defaults describe the 0.92M pilot codec (`codec_d_model` 128,
  `codec_layers` 4, `codec_d_dec` 256, `d_z` 32, `patch` 1), and every real
  run since run-62 is 576/10/8/768 at `d_z` 64, `patch` 3. `resume: !run-62`
  does NOT carry the architecture with it: it names a file, and the file is
  then loaded into whatever model the OTHER inputs built. Run #395 (E-035
  seed 0 re-run) died 90 s in with ~60 lines of `size mismatch for
  encoder.layers.N...: copying a param with shape [576] ... the shape in
  current model is [128]` for exactly this reason. The dispatch is a
  25-field record with no partial-update semantics, so **copy the full
  INPUTS_JSON block out of the log of the run you are replicating** — it is
  printed verbatim near the top of every job — rather than writing the
  handful of fields the experiment is "about".

## 2 · While it runs

- **VERIFY IN THE FIRST MINUTES.** For a resumed or restarted run, check the
  learning rate is what the plan says and is not zero. Cancel if it reads
  otherwise; an hour of a wrong run costs more than a re-dispatch.
- **A queued job against an idle runner is stuck, not slow.** Cancel and
  re-dispatch (this has worked within 90 seconds). Do not restart the boxes
  and lose their warm caches. If a re-dispatch also stalls, stop/start the box
  — that clears a wedged runner and keeps the disk.
- **Watch `gpu_util`.** It is the only signal that says "this is running on the
  wrong device" — four eval scripts silently embedded a 40.7M codec on CPU.
- **Report measurements, not intentions.** "Curves will appear shortly" said
  twice without reading the branch is a guess wearing a fact's clothes.
- **SCREENSHOT THE STATUS PAGE at every monitoring wake-up**, and look at it:
  `node scripts/status_shot.mjs --out /tmp/status.png`. It captures the live
  branches and plans and renders the deployed page at a phone viewport, so
  what you check is what Chris sees. Standing rule, requested 2026-08-10 after
  I twice described a dashboard state that was not real — once because the
  plan preview rendered on a branch nobody is ever on, once because a stale
  record from a previous run was being charted as the current one. Both times
  I reported what the code should produce instead of what the page produces. A
  screenshot is the one check that cannot be fooled by my model of the code.
  It also prints PAGE ERRORS, which is how a blanked section gets noticed
  before a human finds it.

## 3 · Reporting a result

- **Numbers come from `probe_kfold.py`** — the year-blocked k-fold. Anything
  else (in-training light probe, a 36-month split) must be labelled as such.
- **At pentad/daily cadence, spatially POOLED read-outs are distrusted and
  the HEAD probe is primary** (Chris, 2026-08-18: "I don't trust ridge or
  mlp. We should not pool spatially. Let's always look at head").
  probe_kfold's ridge reads `Z.mean(1)` over the ~265-pixel 26.5°N section —
  and geostrophic transport is the east-minus-west contrast ACROSS that
  line, which a mean annihilates. So: pass `head_probe: "true"` on every
  eval dispatch at the new cadences and report `probe_head` (unpooled
  cross-attention over section tokens) next to its raw-3x3 control as the
  headline; keep the pooled ridge as the comparable-to-history number, never
  as the verdict. KNOWN GAP, measured 2026-08-18: stage 2's own transport
  number (`rapid_probe_kfold`) is ALSO section-pooled — temporal.py:2018
  does `hid[:, -1].mean(0)` — so even the big temporal heads are read
  through a spatial mean at the last step. Upgrading that read-out to a
  learned pool (probe_head's mechanism) is an open lever, to be changed as
  its own experiment, never silently.
- **REPLICATES, NOT ARMS. A stage-2 number at n = 1 means nothing.** Measured
  2026-08-11 (E-010): three seeds at one fixed configuration span **0.245** on
  the RAPID head k-fold — sd 0.123 — while the forecast objective those same
  runs optimise reproduces to sd 0.0017. The optimisation is stable; the
  240-month probe with ~9 effective DOF is not. So two configurations
  differing by less than ~0.25 are indistinguishable at one seed each, which
  describes every stage-2 comparison in EXPERIMENTS.md before E-010 — and it
  is how a +0.28 artefact (E-005) survived four months. Before dispatching a
  sweep, ask what its noise floor is; if nobody has measured it, that is the
  experiment.
- **A number without its baseline is not a result.** Wind-only ridge: 0.531 at
  1°, 0.568 at quarter degree.
- **Comparing two probes needs a PAIRED test** (`scripts/paired_probe.py`), not
  two overlapping intervals — they share folds, months and most of their error.
- **Record negative and null results with the same care as positive ones.**
  Most entries in EXPERIMENTS.md are null and they are why the programme knows
  where its bottleneck is.
- **Record the COST too.** E-008 spent three dead dispatches and ~2.5 h of GPU
  before a single training step; an entry showing only the successful run makes
  the answer look cheaper than it was.
- **A rerun is not a resample.** Before dispatching a repeat, check the code
  has a knob for the thing you intend to vary.
- **Every link you post is a MARKDOWN link, one per line** — `[label](url)`,
  never a bare URL and never several joined by `·` on one line (root
  `CLAUDE.md` §0b; Chris could not open the E-022 plan on 2026-08-13 because
  of exactly that). This bites here more than anywhere: a result report is
  mostly links — the plan, EXPERIMENTS.md, the probe archive, the status
  page — and an unclickable link is a result nobody can check.

## 3b · Replication is bought where variance lives, not everywhere

Standing rule, Chris 2026-08-19: *"I don't think we need to runs for every
experiment. At least when two experiments seem to agree a lot during our
experience and the confidence intervals seem small (let's quantify them using
past data given exp scale)."*

§3 says **REPLICATES, NOT ARMS — a stage-2 number at n = 1 means nothing.**
That sentence was written on 2026-08-11 out of exactly one measurement: the
RAPID head k-fold, on a 1.8M head at 6,000 steps, where three seeds at a FIXED
configuration spanned 0.245. It is still exactly right THERE. It is wrong as a
blanket law, because seed spread is not a property of this programme — it is a
property of **metric × scale**, and the archive now spans two orders of
magnitude of it. The table below is that archive, mined from `ml/EXPERIMENTS.md`,
`ml/LEADERBOARD.md` and every `probes-*.json` on `ml-metrics`. **The rule's
authority is the table. When a new replicate lands, the table is extended in
the same commit as the result** — a rule that stops being re-measured is an
assertion.

### (a) What the record has actually measured

Corridor AUC is recomputed to five decimals from each head's twelve archived
per-horizon `msss_clim` values (`horizon_auc` is their mean; the stored field
is rounded to three), so the pair deltas below are not three-decimal artefacts.
"Pooled sd" is the within-configuration sd pooled over every replicate group at
that tier, with its degrees of freedom stated.

| metric | scale — head params · steps · what is scored | replicates in the record | measured spread |
|---|---|---|---|
| rolled **corridor AUC** | **xl tier**: 205–217M head, 60k–200k steps, 12-month roll over the 29,627 corridor pixels | **5 pairs + 1 triple** — E-028 xl55, E-031 xl89, E-032 xl144, E-035 xl233, E-036 zn×xl144, E-037 zn×xl233 | pair \|Δ\| **0.0020, 0.0023, 0.0033, 0.0038, 0.0051**; xl55 triple range 0.0011. Pooled sd **0.0021** (7 dof), 95% upper bound **0.0037** |
| rolled corridor AUC | **88M tier**: 768×12, 60k–200k | 4 triples + 2 pairs (E-027 big34/big55, E-029 r222/znoise/sun89 60k & 200k) | ranges **0.0011–0.0150**; pooled sd **0.0056** (10 dof) |
| rolled corridor AUC | **34M tier**: 576×8 / 32.0M, 60k — the E-017/022/023/026/027 geometry arms | 13 triples + 1 pair, 14 configurations | ranges to **0.0224**; pooled sd **0.0070** (27 dof) |
| **transport band r**, on the SAME xl checkpoints | 205–217M, three held-out years | 2 pairs (xl89, xl144) | **0.05 and 0.07** — 10–30× the corridor spread, read off the same checkpoints in the same files |
| rolled corridor AUC, **`_trainlon` scope** | xl tier, xl233 pair (#417, 2026-08-20) | 1 pair | \|Δ\| **0.00075** — a THIRD of the same pair's blended 0.00200, and the tightest reading of any pair in the record. Scoring only pixels the model was trained on removes a variance source, it does not add one |
| rolled corridor AUC, **`_holdlon` scope** | xl tier, xl233 pair (#417, 2026-08-20) | 1 pair | \|Δ\| **0.00958** — **4.6× the tier's pooled sd** and larger than the largest blended pair delta ever measured here (0.0051). **A `_holdlon` number must not be quoted at n = 1**: it scores extrapolation into a training hole, which is the least reproducible thing this programme does |
| **transport band r**, three-band split | 205–217M, xl233 pair (#417) — a THIRD pair for the row above | 3 pairs total | h1-3 **0.042** · h4-6 **0.066** · h7-12 **0.069**; confirms the 0.05–0.07 band-r regime rather than narrowing it |
| **forecast ratio** (z-mse / persistence) | 1.8M · 6k | 2 triples (E-010) | sd **0.0017** |
| forecast ratio | 205M · 60k | 1 triple (E-028 xl55) | spread **0.004** |
| **RAPID head k-fold** (`rapid_probe_kfold`; 240 months ≈ 9 effective DOF) | 1.8M · 6k | 2 triples (E-010) | U=1 range **0.245**, sd **0.123**; U=4 sd 0.040 |
| RAPID head k-fold | 1.8–10.7M · 60k | 5 triples (E-012, E-013b, E-014, E-015, E-016) | per-arm sd 0.024–0.150; pooled **0.095** (10 dof) |
| **codec head probe** (frozen embeddings, 240 months) | 0.92M codec · 40k · 1° global tensor | 1 codec-seed pair (patch24, #18 / #43) | attention head **0.036**, pooled ridge **0.012**, raw-pixel control **0.049** |
| anything at **pentad or daily cadence** | E-038a/b/c, E-042, the family-4 codecs | **none — every arm at the new cadences is n = 1** | **UNMEASURED** |

Be honest about the n. Three pairs at 0.002–0.003 is three pairs, not a
distribution; the xl row is five pairs and one triple, seven degrees of freedom
in total, and that is what licenses "the seed sd at this tier is 0.0021 and
very unlikely to exceed 0.0037" — a bound, not a law. It says nothing whatever
about the tail at a tier nobody has replicated.

**Two things in the archive that look like replication and are not.**

- **Protocol determinism.** The `e017_u1_s0` gate head has been re-rolled in
  **eighteen** separate eval runs (#228 … #413) and returns gate AUC **0.643**,
  corridor **0.589** and window **0.622** every single time. `probe_kfold` over
  the f3_anchor41M codec on the pentad tensor returns rapid r **0.660**, rmse
  2.97 and an identical CI in #390, #392, #397 and #406. #116 reproduced the
  whole probe ladder bit-for-bit (E-003b) because `probe_head.py` had no seed
  argument at all — its three per-fold seeds were the literal tuple `(0,1,2)`.
  **Re-running a fixed checkpoint through a fixed protocol measures the
  PROTOCOL.** It is a first-class integrity check — it is the certificate that
  makes an eval wave readable at all — and it is not a replicate. Do not put it
  in the spread column.
- **The box effect.** Two boxes that had each built their own tensor moved the
  head k-fold by **0.041** and the 36-month split by **0.111** at a fixed seed
  (E-008 §"The box effect, finally MEASURED"; #131 against #140). That is an
  environment term, not a seed term, and it is why cross-box arms are not
  pooled. Published `Z` removed the cause; the number stays on the record as
  the size of what a stray environment difference can buy.

### (b) The rule

**One seed is enough** when all three hold:

1. the result is scored by **rolled corridor AUC**;
2. at the **xl tier or above** — ≥205M head, ≥60k steps, frozen f3 anchor
   codec, monthly `family3_na025` tensor: the one configuration the pairs above
   actually measure;
3. and the claimed effect is **≥ 0.025** corridor AUC.

**Where 0.025 comes from, and why it is not 0.015.** Two derivations, both out
of the table, and they agree. Five times the LARGEST pair delta ever measured
at the tier (0.0051, E-032 xl144) is 0.0255. And the quantity actually at risk
is a DIFFERENCE of two single-seed numbers, whose sd is √2 × the tier sd; at
the 95% upper bound on that sd (0.0037) that is 0.0052, and five of those is
0.0262. Call it **0.025**. The bar is anchored on the largest observed delta
rather than the median one (0.0033) because a rule calibrated to the median is
calibrated to the lucky half of the record.

Read back against the record, 0.025 is the bar that behaves. It admits every
effect this programme has replicated — input noise **+0.045 / +0.050**
(E-036 / E-037), capacity **+0.042** across 88M → 205M (E-028) — and it refuses
every effect the log later had to hedge: width **+0.005**/rung, the step
budget's **−0.0006**, xl233's **−0.0026**. All three of those needed their
pairs, and all three got them.

**Two seeds remain mandatory.** No exceptions, and "the direction is obvious"
is not one:

- **Any probe-scored claim.** The head k-fold's regime is **0.036–0.245**, ten
  to a hundred times the corridor's. No probe number is readable at n = 1, at
  any scale this programme has run.
- **Any new metric, cadence, tensor, codec or scale tier with no measured
  pair. The first result at a tier buys its own replication.** Every pentad
  and daily arm is in this class today. So is any run on a new codec: all six
  xl pairs share one frozen 40.7M codec and one monthly tensor, and a band is
  warranted only where it was measured.
- **Any number that will be quoted as a headline in the paper**, whatever its
  size. The paper's own voice already says single-seed head numbers *"should
  not be quoted anywhere, including by us"*.
- **Any claim that an effect is ZERO, or that an axis is CLOSED.** A null is a
  statement ABOUT the noise band and cannot be made from one draw out of it. A
  closure is the most fragile claim this programme makes — both of the ones it
  has had to reverse (E-010 on capacity, E-032 on width) were nulls read off a
  scoreboard that could not resolve the effect, and §"HARVEST #344–#347" states
  the general form: *a settled negative is settled only on the scoreboard that
  settled it*.

**A single-seed result inside its tier's band is written as a CONSISTENCY,
never as a level.** This is already the paper's voice; it is now the log's.

- ✅ `consistent with zero at n = 1, against a tier spread of ±0.005`
- ✅ `consistent with xl144's 0.6781 — a −0.0026 difference is inside the band`
- ✅ `+0.050 at n = 1, ten times the tier's largest measured pair delta`
- ❌ `xl233 rolls at 0.673` — a level, from one seed. This was the state on
  2026-08-18 and the log said so at the time (*"that reading is currently
  carried by an n = 1 number, which ml/CLAUDE.md §3 says means nothing"*);
  #396 and #413 were dispatched for exactly that reason and the pair now reads
  **0.67492 / 0.67292**.
- ❌ `xl233 is 0.005 below xl144` — a difference quoted from one draw each,
  which is the E-005 failure mode with a newer metric.

### (c) Why the rule exists, and what it costs both ways

**GPU-hours are the scarce resource, and the replicate is priced.** Measured on
this wave: **#396 (E-035 seed-0 roll-forward, 200k steps at 217M) cost 15.6 h
and ~$4.6**, and its share of the eval — one more xl233 head inside an existing
`sroll:` run — is ~3–5 h and ~$1.0–1.5. A second seed at xl scale is therefore
**about $6 and about a day of a rented box**, against a measured spread of
0.002–0.005 on the metric it replicates. That is the trade the rule declines.

At probe scale the same money is not optional, **because that noise once
manufactured a result and the programme believed it.** E-005 reported
**+0.28** on the AMOC probe for autoregressive unroll, from #88 (U=1, 0.173)
against #93 (U=4, 0.449) — n = 1 each, different dispatches months apart,
scored on a 36-month single split. E-009 re-scored the axis under the
year-blocked k-fold and found U=1 ABOVE U=2 by 0.178 while the old split
ordered the same two runs the other way. E-010 then measured the thing nobody
had: three seeds at a FIXED U=1 span **0.245**, sd 0.123 — while the forecast
objective those same runs optimise reproduces to **sd 0.0017**. The noise floor
was larger than E-009's gap and the same size as the original claim. **E-005 is
dead, not withdrawn**, one draw against another, and the unroll axis went with
it.

Which is the rule in one line: **the replicate is bought where the variance
is, and the variance is in the READ-OUT, not in the training.**

---

## 4 · Working principles (written after the night of 2026-08-09)

That session burned about six hours and a dozen dispatches and produced one
scientific result. Everything else was self-inflicted. These are the rules
extracted from it; they are ordered by how much they would have saved.

**On building.**

1. **Prefer the formulation that removes a failure mode over the one that
   guards against it.** The joint loss got three revisions of increasing
   cleverness — a step-0 constant, a scale-free ratio, a twin reference
   trained alongside — each *policing* a degeneracy that a different choice of
   measurement space *abolishes*. When you are adding a correction to a
   correction, the earlier choice is wrong, not under-specified. Stop and
   change it.
2. **Normalise by properties of the DATA, never by properties of the MODEL.**
   A denominator the model can influence is a term in the objective, and it
   will be optimised. Variance of an observed field is fixed and ungameable;
   a frozen checkpoint's loss is not, once what was frozen is what you are
   training. This one sentence covers the shrinkage degeneracy, the detached
   denominator, and both circular reference constants.
3. **Keep diagnostics out of the objective.** "Am I still as good as the model
   I started from?" is a question to LOG, not to optimise. Smuggling it into
   the loss is what made one experiment depend on a constant hand-copied from
   another, and then on that other experiment's arbitrary stopping point.
4. **Sequenced experiments are a smell.** If B needs a number from A, ask
   whether B can measure it itself. Interdependence costs A's wall-clock plus
   a constant that silently goes stale.
5. **For a change to an objective, write the algebra before the code.** The
   detached-denominator bug was a two-line sympy check away, and that check
   is what finally settled it. For a loss, the gradient IS the specification;
   the code is a transcription.

**On verifying.**

6. **A step that can fail silently will.** Every `|| true`, `2>/dev/null` and
   `continue-on-error` must say why it gave up. Best-effort is a promise about
   *delivery*, not about *reporting*. Four separate steps in one night
   reported success while doing nothing: a release seed with a mis-braced
   variable, the same seed calling a CLI the boxes do not have, a live-metrics
   publisher with no token, and a dispatch that silently chose a CPU box.
7. **Assert the EFFECT, not the invocation.** The seed step printed "trying
   release asset …" and downloaded nothing for three runs. A log line proves a
   line of code ran; it proves nothing about the world.
8. **Exercise the code path on a toy before spending the expensive
   resource.** A synthetic 8×9×5 tensor and a 1-layer codec took five minutes
   and exercised three loss modes end to end. Any hour of GPU on a path that
   has never executed is a coin flip. `ml/train_joint.py --smoke` against a
   generated npz is the pattern.
9. **Build an invariant with an EXACT expected value and make the job refuse
   if it fails.** "r_fore must read exactly 1.000000 at step 1, because both
   branches are identical until the first update" is worth more than any
   amount of careful reading. Prefer exact identities to threshold checks —
   thresholds are the tripwires that killed healthy runs (#86, #91).
9b. **A degeneracy you can NAME is one you must close or measure — never one
    you may rank as improbable.** While designing the scale-free denominator I
    wrote down the second cheat (inflate the persistence baseline instead of
    shrinking z) and judged it "a real possible cheat but far less trivial
    than pure scaling. Worth noting, not worth blocking." It was the dominant
    direction and arrived faster than the one being fixed. If you can describe
    the exploit in a sentence, the model can find it in a thousand steps.
10. **Instrument the quantity that DISTINGUISHES the stories, not the one
    that is easy to plot — and guard BOTH directions.** Reconstruction-on-a-fixed-batch looked healthy
    through a 40× embedding collapse because it was structurally blind to it.
    Ask: *what would look identical whether this works or fails?* — and
    measure that. `z_shrink` turned the next occurrence into a ten-minute
    diagnosis instead of a retracted result — and then failed on its own
    terms: it was written for CONTRACTION and coloured red only above 1.2, so
    the 250x EXPANSION that followed rendered in grey and looked ordinary. A
    one-sided guard on a two-sided quantity is not a guard. Flag on
    |log(ratio)|.

**On deciding and reporting.**

11. **Distinguish "it is running" from "it can answer the question."** Several
    runs were technically healthy and could not, by construction, test their
    own hypothesis: a 95/5 gradient split, a reference that put the forecast
    term permanently below the reconstruction term. Before dispatch, state
    what result would falsify the hypothesis and check the configuration can
    produce it.
12. **Report measurements, not intentions.** "Curves will appear shortly" was
    said twice without reading the branch. An unchecked status claim is a
    guess wearing a fact's clothes.
13. **Prefer one clean run to five relaunches.** Each relaunch was locally
    justified; collectively they meant nothing finished. When the same
    artefact is relaunched twice, stop spending and fix the whole class of
    problem first. An idle GPU at $0.27/h is far cheaper than a confident
    wrong experiment.
14. **A queued job against an idle runner is stuck, not slow.** Cancel and
    re-dispatch (see the queue-stall note in Part 2); do not restart the boxes
    and lose their warm caches.

---

## 5 · Principles added 2026-08-10

The day E-008 took three dispatches to start. Same spirit as §4, different
failures.

15. **Verify the ARTEFACT, not the intention.** The E-008 design said the run
    "reloads the 60k model, optimiser moments and RNG stream". No published
    head has ever carried optimiser state. The claim was written from what the
    checkpoint was assumed to contain, and cost 93 minutes of embedding to
    disprove. Open the file.
16. **A guard can be right and in the wrong place.** The refusal that caught
    it was correct and fired after the expensive part. A precondition that
    depends only on the inputs must be checked while the inputs are all it has
    cost you.
17. **Only a DEFINITE answer may be fatal.** GitHub runs `run:` blocks under
    `bash -e`, so a non-zero exit from a precheck aborts the step and takes the
    whole probe ladder with it. Right for "this cannot work"; wrong for a torch
    hiccup. A check that can cost more than the thing it protects is the same
    error one level up.
18. **Size a guard from the allocation it guards.** The disk check fired below
    8 GB against a single 10.4 GiB write, so it could not fire in time by
    construction. Where the size is computable, compute it — and put the check
    where those numbers are in scope, not in a shell step that can only guess.
19. **Let a check CHOOSE a resource, not merely permit or refuse one.** The
    embedding memmapped to a 50 GB disk on a box with 126 GB of RAM, because
    the code was written for a 7 GB box. When one resource is exhausted and
    another is not, use the other and say so.
20. **Publish a shared artefact when it EXISTS, not when the job ends.** The
    embed-cache push sat after `wait $S2_PID`, i.e. sixteen hours after the
    cache was finished — so "wait for the first job, then start the second"
    described nothing that could happen. A dependency graph over jobs that do
    not publish until they finish is just a slower sequence.
21. **Flush, THEN mark.** A resumable artefact's progress marker must be
    written after the data it describes, so it can only under-claim. An
    over-claiming marker makes the next run skip work that was never done —
    real numbers, wrong months, no symptom.
22. **Never write NaN into a results file.** Stop instead. A results file full
    of NaN is loud enough to notice and quiet enough to misattribute: the
    all-NaN sequence probe was blamed on the probe twice before anyone looked
    at the mask.
23. **A display only helps if its render condition holds during the window it
    is for.** The planned-schedule preview drew only when the live branch had
    no metrics file — true for a few seconds of a sixteen-hour run, false
    every time a human looked. Test the state the user will actually be in.
24. **Do not leave a stale reference table in the log.** A planned-LR table
    that no longer matches the run is worse than none: it gets checked, it
    matches nothing, and the run takes the blame for the document.

---

## 6 · Security posture of the rented GPU boxes

The Vast boxes are strangers' machines. The host has root on the physical
machine, so **treat everything on a box as readable by whoever owns it**.
That single assumption decides everything below.

**What is on a box.** The workflow's automatic `GITHUB_TOKEN` (an env var for
the job's lifetime, repo-scoped, `contents: write` per the `permissions:`
block, revoked by GitHub when the job ends); the Actions runner's registration
credential under `/opt/runner`; and public data plus model checkpoints, none
of it confidential. The runner runs as root inside its container
(`RUNNER_ALLOW_RUNASROOT=1`).

**What is NOT, and must never be.** The user's PAT (`/home/claude/.gh_pat`
lives in the session sandbox), the Vast API key, and the CMEMS credentials.
Nothing that outlives a job and nothing that reaches another repo or account.

**The registration is a bigger exposure than the token.** The job token dies
with the job. The registration persists on disk, survives stop/start, and is
NOT `--ephemeral` — so anyone who lifts it can act as a runner for this repo
indefinitely, receiving jobs and minting a fresh `GITHUB_TOKEN` each time.
Mitigations, cheapest first: pass `--ephemeral` when creating a box (one
registration per job); remove the runners from the repo
(`scripts/fleet_rm_runner.mjs`) when the fleet will be idle for a long stretch
— note this breaks the cheap stop/start loop, since a started box reconnects
with a credential GitHub no longer honours and the onstart script skips
re-registration when `/opt/runner/.runner` exists.

**The one line of defence that must never move.** Self-hosted runners on a
PUBLIC repo are the classic dangerous configuration: a fork's pull request can
execute arbitrary code on your hardware. We are safe only because
`ml-train.yml` is `workflow_dispatch`-only and dispatch requires repo write
access. `pages.yml` does carry `pull_request`, but runs on `ubuntu-latest`.
**Never add `pull_request`, `issue_comment`, `workflow_run` or `schedule` to
ml-train.yml, and never point another triggered workflow at `runs-on: gpu`.**
That change would look harmless in review and would hand code execution on the
boxes, plus the job token, to anyone who opens a PR.

**Blast radius if a job token does leak.** Repo-scoped `contents: write`: push
to any branch including main, delete branches, cut releases. Not other repos,
not the account. Keep `permissions:` at `contents: write` and add nothing;
`contents` is needed only for the `ml-live-*` side channel and the `ml-metrics`
archive.

---

## 7 · Domain lore (hard-won; do not relearn)

### The control plane

- **`workflow_dispatch` allows at most 25 inputs, and a 26th breaks the WHOLE
  workflow.** `ml-train.yml` sits exactly at the ceiling. Exceeding it does not
  fail gracefully — GitHub refuses to parse the file, every dispatch 422s, and
  the Actions list shows a failed run named after the file path with no jobs,
  which reads like a broken runner. Encode new knobs into existing inputs:
  `window` carries the loss mode, `unroll:N`, `resume2:<tag>[@lr]` and
  `warm2:<tag>[@lr]`; `lr_floor` carries the joint codec LR. **Count the inputs
  before pushing a workflow edit.**
- **`runner` defaults to `gpu`.** Omitting it once sent a run to a free
  4-core GitHub box where the install step picks the CPU torch wheel and a 40M
  codec "trains". Nothing in the output says wrong-hardware; the only tell is
  the runner name.
- **A queued job can wedge.** Runs have sat queued for 22 minutes while the
  API reported the runners online and idle. Cancel and re-dispatch — that has
  worked within 90 seconds. Check `runner_name`/`status` on the JOBS endpoint,
  not the run-level status, which lags.
- **`bash -e`** governs every `run:` block, so any non-zero exit aborts the
  rest of that step. This is why a precheck must only fail on a definite
  answer (§5.17).

### The boxes

- **A Vast instance's runner NAME lives in its `onstart` script and nowhere
  else.** The names (`gpu-box-45318655`) are stale instance IDs and do not
  match current IDs; every box carries the same `label: earth-runner`. Recover
  the mapping with `(i.onstart||"").match(/--name[= ]"?([^"\s\\]+)/)` over
  `/api/v1/instances/` before stopping anything, or you will stop the box that
  is mid-run.
- **A Vast disk CANNOT be resized, and the API lies about it.** `PUT
  /instances/<id>/` with `{"disk": 80}` and with `{"disk_space": 80}` both
  return `200 {"success": true}` and change nothing — measured on two stopped
  instances, v0 and v1. `scripts/gpu_box.mjs resize` refuses with that
  measurement rather than issuing a call that appears to work. To get more
  disk, create a box with a larger `DISK` and destroy the old one; the data
  cache re-seeds from `data-cache-v1`.
- **50 GB is a real design constraint**: a ~15 GB torch image, ~11 GB of
  tensors and a 5.2 GiB embedding cache leave little room. At float32 the
  cache did not fit at all, which produced a delete-rebuild-delete treadmill.
  It bit again on 2026-08-11 in a nastier shape: the disk crept to 50/50
  mid-queue, hygiene freed the published Z (correctly — it is re-pullable),
  the disk STAYED full from something else, the re-pull refused for lack of
  headroom, and every run fell back to an ~80-minute in-RAM rebuild it could
  not persist — while going metrics-blind, because every disk write
  (metrics.jsonl appends, live-branch pushes, checkpoint mirrors) fails
  silently behind its own best-effort guard. **A full-disk box computes
  fine and reports nothing**; check Vast's `disk_usage` before trusting a
  silence. Default `DISK` is now 100 GB (`gpu_box.mjs`).
- **Fleet policy since 2026-08-11 (Chris): experiments run PINNED to the
  healthy box; a box with a sick disk is for investigation only** — "a box
  without disk is almost impossible to debug; usually it crashes for some
  unknown reason." Pin with the `runner` input: the runner's NAME is an
  implicit label (`runner: gpu-box-40623952`). Current roles: experiments
  on `gpu-box-40623952` (Vast 47483091, 100 GB); `gpu-box-35586926`
  (47171781, 50 GB, disk full) is the debug box pending the `disktriage`
  verdict. The `disktriage` window mode exists because the sandbox cannot
  SSH a running Vast box (port blocked; the execute API refuses on running
  instances) — a job's own eyes are the only eyes.
- **Cross-box comparability is a PULL, not a hope**: since 2026-08-11 the
  workflow seeds the sha-pinned tensor (`adcbe700…`) from `data-cache-v1`
  and verifies it before `build_family3.py` (whose recipe guard then skips
  the build). A box only builds its own tensor when the pull fails, loudly,
  and provenance records what it really used.
- **Stopping is the cheap idle state.** It keeps the disk (data cache,
  checkpoint mirrors) and the runner registration, dropping a box from ~$0.27/h
  to storage only. `start` can return `resources_unavailable` and queue the
  state change — it may come up minutes later, so check before renting more.
- **Nothing that outlives a job may be stored on a box** (§6).

### Failure signatures worth recognising instantly

- **A GREEN run with no `temporal.json` in its probe archive = the trainer
  died and nothing noticed.** temporal.py runs BACKGROUNDED behind
  best-effort guards, so a `CUDA error: unspecified launch failure` (or any
  trainer death) does not colour the run. Measured on #196 (2026-08-11): a
  lemon box's GPU killed stage 2 six minutes in, every guard said
  "continuing", the run concluded success. The archive's file list is the
  truth; the run's colour is not. An anomalously FAST ms/step is the same
  sick-GPU story from the other side (19.7 ms/st for a U=4 arm, vs 62 on a
  healthy box). Fresh Vast hosts can simply be broken — destroy and re-rent
  on a different machine; do not debug a lemon.

- **`"$TAG__pixelmae.pt"`** expands the variable `$TAG__pixelmae` — underscores
  are legal in a bash identifier. **Always brace a variable followed by `_`, a
  letter or a digit**: `"${TAG}__pixelmae.pt"`.
- **The boxes have no `gh` CLI, and no `node` either.** Fetch release assets
  with `curl -fsSL https://github.com/$REPO/releases/download/$TAG/$ASSET`
  (public repo, no auth). Anything a workflow step runs on a box must be
  bash or **Python** — `scripts/archive_probes.mjs` was wired in on
  2026-08-10 and died with `node: command not found` on every job for four
  hours, warning quietly while the jobs went green, so not one probe result
  was archived by it. `scripts/archive_probes.py` is the version that runs.
  The SANDBOX has the opposite constraint — node writes to api.github.com
  fine, python is blocked by the egress proxy — which is why both exist.
- **A failing best-effort step can still kill the ones after it.** `bash -e`
  means a non-zero exit aborts the whole `run:` block, so "best effort" has
  to be written at the CALLER (`|| echo "::warning::…"`), never faked by the
  callee returning 0. On 2026-08-10 `embed_cache_sync.py push` was correctly
  changed to exit 1 on failure and the bare call site was not updated: #131's
  5.2 GiB upload found no room, exited 1, and took `probe_kfold.py` and
  `dip_check.py` down with it. The run reported success.
- **`curl --data-binary "@file"` BUFFERS THE WHOLE FILE.** Use `-T file`,
  which streams. Measured on a 300 MiB file: 226 MiB peak RSS against 10 MiB,
  both returning 201. This is why `embed-cache-v1` sat at zero assets for a
  full day of runs — a 1.5 GiB chunk died with `curl: option --data-binary:
  out of memory` on the FIRST chunk, every time, since the day the feature
  was written. Three other real bugs were found and fixed at that same call
  site first (an always-zero exit code, an ENOSPC that filled a box, a caller
  with no `|| true`); none of them was the reason it had never worked. **Read
  the failing line before fixing the third thing around it.**
- **`2>/dev/null` on a command whose failure you are branching on** hides the
  one line that explains the branch.
- **A channel that starts later than the tensor** makes a month-0 mask empty,
  and `mean` over an empty section is NaN, not an error. See
  `docs/ML_BASICS.md` §3.
- **`CosineAnnealingLR.load_state_dict`** restores the parent's `T_max`, so a
  reloaded schedule asked for a larger total returns **lr = 0.0**. See
  `docs/ML_BASICS.md` §9; `--lr-schedule invsqrt` removes the whole class.
- **No stage-2 head published before 2026-08-10 can be continued** — all are
  `{args, model}`. Use `window: warm2:<tag>[@lr]` with `temporal_steps` as the
  EXTRA steps. Snapshots written since carry `opt`/`sched`/`step`.

### Artefacts and where they live

| artefact | where | keyed by |
|---|---|---|
| codec checkpoints | `model-checkpoints-v1` release | tag |
| stage-2 heads | `model-checkpoints-v1`, `run-<n>-temporal-latest.pt` every ~30 min | run number |
| tensors | `data-cache-v1` release, chunked | recipe |
| embedding cache `Z` | `embed-cache-v1` release, 1.5 GiB chunks | **codec weight hash** |
| live metrics | `ml-live-<n>` orphan branch (force-pushed, deleted at end) | run number |
| archived metrics | `ml-metrics` branch (additive) | run number |

---

## 8 · Deferred / open follow-ups

- ~~**Build E-006**~~ — done 2026-08-10: `--loss-mode data` in
  `ml/train_joint.py`, with the gauge invariance checked on the real model
  (`tests/test_e006_gauge.py`: data-space loss ratio 1.000000, z-space
  16.000000 under a ×4 rescale) and an end-to-end CPU smoke
  (`tests/test_e006_smoke.py`). What remains is a DISPATCH decision — budget,
  and the control, which should be the frozen codec rather than another joint
  mode, since the question is whether joint training beats not doing it.
- **Split the pipeline into three jobs** (embed / train A / train B) — designed
  in `docs/INFRASTRUCTURE.md` §6b, now possible because the cache publishes on
  existence.
- ~~**Decide on `--lr-schedule invsqrt`** as the default~~ — settled by
  E-008d/e (2026-08-11): `expdecay` for horizon-freedom, and the terminal
  taper to zero is worth 0.16% — real, negligible, keep it. The schedule
  question is closed by being uninteresting.
- **Run `scripts/paired_probe.py`** over the head and its raw-3×3 control once
  a run has written the per-month arrays, and only then decide whether the
  +0.034 gap is quotable. Draw a genuine second seed with `--seed-base 3`.
- ~~**Re-score #88/#93 through `probe_kfold`**~~ — **impossible, struck
  2026-08-10.** `probe_kfold` scores the frozen codec, which #88 and #93
  share, so it returns the same number for both by construction. Asking for
  it is what sent E-009 out with an instrument that could not move with its
  own variable. The replacement is `rapid_probe_kfold` (`temporal.json`), and
  E-009's #131–#134 re-measure the whole U axis on it. See
  `docs/ML_BASICS.md` §5b.
- ~~**Unroll U=2 and U=8.**~~ — **closed 2026-08-11 by E-010.** Three seeds at
  each of U=1 and U=4: the AMOC probe differs by +0.023 (t = 0.31, one tenth
  of the seed range), and the forecast is 29.7% WORSE at U=4 (t = 99). Unroll
  costs skill and buys nothing on transport. `--unroll` should default to 1.
  Filling in U=2 and U=8 would be measuring a difference already shown to be
  absent.
- **The parameter-bottleneck question** — needs a from-scratch run at larger
  width; deliberately deferred.

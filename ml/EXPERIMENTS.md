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
6. **Every entry states the four scale numbers** (Chris, 2026-08-11):
   **parameters · batch size · steps · data points** (for stage 2: train
   windows in the pool; for a codec: pixels × months sampled). Since
   2026-08-11 the trainer writes them itself as `temporal.json:scale` —
   quote that block, never a hand-carried number. Reference values:
   stage-2 192/4 = 1,822,144 params (+37,056 with three direct heads);
   384/6 = 10,732,096; the quarter-degree U=1 pool ≈ 26.1–28.1 M windows at
   batch 256 (temporal.py's default — the workflow's --batch 512 goes to
   the CODEC trainer; measured by the source-written scale block on #202,
   which caught this very file hand-carrying 512).

Baselines, for reference. Wind-stress-only ridge in our own protocol:
**0.531** on the 1° tensors, **0.568** on the quarter-degree tensor. RAPID
monthly σ = 2.79 Sv, n = 240, ~68 effective DOF (~9 after an 18-month
low-pass).

---

<a id="e-058"></a>
## E-058 · Per-channel rolled skill, so a roll can be read for SST as well as for AMOC — RUNG 1 BUILT + CERTIFIED (no GPU spent)

TL;DR — every skill number this programme has ever published from a roll was
pooled over all 40 channels at once, so "does the predicted embedding get
sea-surface temperature right?" was a question the artefacts could not answer.
This adds the per-channel decomposition beside the pooled rows, changing no
pooled number by a single byte. It is rung 1 of Chris's multi-target
directive (2026-08-28): *"I'd love to predict ocean surface temperature as a
secondary downstream target (next to AMOC). It's important to have multiple
such targets (maybe even a third one) to ensure the embedding representation
is comprehensive and not just AMOC tailored."*

**E-058 rung 1 · per-channel rolled skill · params n/a · stage tooling ·
data family4_na025_pentad_r2 (and every tensor a roll reads) · arch n/a ·
steps×batch none (nothing trains) · resume n/a**

### Why the existing artefacts could not answer it

Two facts, both measured rather than assumed:

- **`chan_skill` is per-HORIZON, not per-channel.** The name is a legacy from
  `rollout.py`, where a "chan" row WAS a horizon. Every row in it is pooled
  over all 84,405 window pixels AND all 40 channels, so `sst` — appended by
  E-042 and last in the tensor — contributed one fortieth of a number nobody
  could decompose.
- **The tensor is z-scored anomalies**, so `mse_c` is `(v_true**2)` and the
  climatology reference IS the zero forecast. `msss_clim` per channel is
  therefore readable directly, with no extra baseline to compute.

### What was built

`ml/rollout_spatial.py`, additive only:

- `new_sums(H, C=None)` optionally allocates a parallel `su["per_chan"]`,
  `{key: [H+1, C]}`, over the same ten sum keys (`SUM_KEYS`).
- `accumulate(...)` fills it **on the same early-exit branch, from the same
  masked `[n_pixels, C]` arrays**, with `.sum(axis=0)` where the pooled body
  takes a scalar sum. The pooled body is byte-for-byte what it was. A mask
  that is not `[n_pixels, C]` raises rather than broadcasting into a wrong
  answer.
- `_skill_rows(su, H)` is the original loop lifted **verbatim**, and is now
  the ONLY arithmetic path — pooled rows and per-channel rows are produced by
  the same code, so the two cannot drift.
- `skill_block(..., chan_names=None)` emits `per_channel` **between `n_px`
  and `horizon_auc`**, deliberately not appended, so that
  `horizon_auc_daymatched`'s "written LAST" invariant — which the day-matched
  byte strip's comma removal depends on — stays true.

### The certificate

`tests/test_per_channel_skill.py` (5 checks, all pass):

- **Pooled purity is BIT-IDENTICAL** — 18,289 bytes identical after removing
  the 18 `per_channel` blocks.
- **Consistency**: the pooled `msss_clim` IS the climatology-mass-weighted
  mean of the per-channel ones, deviation **1.11e-16**.
- **Known-answer**: a channel predicted perfectly reads `msss_clim` **+1.000**
  and a pure-noise channel **−1.025**, while the pooled row containing both
  says **+0.310** — which is the whole point: the pool hides both.

Two pre-existing byte-identity tripwires — `tests/test_roll_monthly_identity.py`
and `tests/test_fgn_roll.py` — failed on the new key, **correctly**: they exist
to force a human decision when the roll payload changes. Both were widened
under their own documented pattern (strip → COUNT → PIN), never relaxed. The
pin is stronger than key-identity in three independent ways, and the third is
the one worth keeping:

1. the channels' `n` sum to the pooled `n` EXACTLY at every horizon (same
   cells, or the two sum sets were accumulated over different pixels);
2. the pooled `msss_clim` lies inside its own channels' `[min, max]` at every
   horizon (a positive-weight mean cannot leave the convex hull of its terms
   — the checkable consequence of the 1.11e-16 relation, whose weights the
   artefact does not publish);
3. on the **corridor** scope every value equals the **E-026b audit block's**
   per-channel decomposition of the same quantity — a decomposition that
   predates E-058 and is accumulated by different code in `aud["ch_m"]` /
   `aud["ch_c"]`. Measured deviation **0.0**.

`tests/test_roll_ring_identity.py` and `tests/test_roll_dump.py` were ALREADY
failing before this diff (verified by `git stash`) and were not touched.

### Cost, and what it does NOT yet do

**No GPU was spent.** Rung 1 is a read-out: the next roll that runs writes the
SST column for free, and no roll needs re-running to get it.

It does **not** make SST a training target, and it does not settle whether the
embedding is AMOC-tailored — it makes the question askable.

### Rungs 2 and 3, AFTER the readiness audit — and the audit RETRACTS both of the first drafts

The two rungs were first written here as "add SST to `head_targets`" and
"Florida Current as the independent third target". A read-only audit of what
is actually plumbed (2026-08-28, 230 archived bundles on `ml-metrics` plus the
builders) **retracts both**, and the retractions are more useful than the
proposals were.

**RETRACTED: the Florida Current is not an independent target.** `probe_kfold`
puts `rapid` at lat 26.5, lon (−80.0, −13.0) and `fc` at lat **26.5**, lon
(−80.5, −78.5) — the SAME grid row, and `section_of` clips both to it. FC's 9
grid points contain 7 ocean cells (`EXPERIMENTS.md`'s own earlier note: "the
Florida Straits lon … 7 cells"; `probe_kfold.py:70` puts it more sharply — "the
Florida Straits are 5 of RAPID's 265 cells"), and every one of them is inside
RAPID's 266. Physically the same: `ml/LEADERBOARD.md:327` records **RAPID =
Florida Current + Ekman + upper-mid-ocean geostrophic** — FC is one of the
three terms RAPID is a sum of. So an embedding tailored to RAPID's section is
BY CONSTRUCTION tailored to FC's. FC remains a good LABELS lever (a cable
record, 2,490 pentads against RAPID's 1,459, and it already reads r 0.375
[0.286, 0.455] pooled on the r2 pentad tensor against a 0.199 wind bar) and it
is a poor COMPREHENSIVENESS lever. The earlier sentence in this section, and
the same claim in the 2026-08-28 handoff, were wrong.

**And the general form, which is the finding worth keeping: no transport
target can answer whether the embedding is AMOC-tailored, because all five of
them ARE AMOC.** `fc`, `move` and `osnap` are AMOC arrays at different
latitudes; `samba` is one too, and is outside the tensor window (34.5°S
against a 0–70°N grid) at any price. Chris's directive asked for a different
PHYSICAL QUANTITY, and that is exactly what SST is — which is why rung 1 is
the substantive answer and the transport targets are not.

**RETRACTED as written: "add SST to `head_targets`".** Two obstacles, one
mechanical and one fatal. Mechanically, `target_series` needs a `[n, 2]`
(axis-row, scalar) series under a `truth_*` key with ≥48 samples plus a
lat/lon section; no `truth_sst` exists anywhere in the repo, so this is a
DATA BUILD, not a recipe key — and SST has no natural zonal section, so the
choice of box would drive the answer. Fatally: the k-fold ladder and the head
are both CONTEMPORANEOUS read-outs, and `sst` is channel 40 — an encoder
**input** — of the r2 tensor. "Can a ridge read an SST index off an embedding
of a field that includes SST" is close to asking whether the autoencoder
autoencodes; a high number would be uninformative and, worse, quotable.

**What replaces them.**

- **Rung 2 (the honest version): the SST probe belongs on an r1 codec, which
  never saw SST.** Same machinery, one variable moved, and then a high number
  IS evidence that the representation carries surface thermal structure it was
  never shown. That is a real experiment; the r2 version is a pipeline check
  wearing its clothes. Note it is still gated on building `truth_sst`.
- **Rung 3: a different physical quantity, not a fourth AMOC array.**
  Upper-700 m ocean heat content from the `rg_t` channels is the leading
  candidate — a stored, slow quantity where transport is a fast one, and
  already present in the tensor.
- **Free, and worth taking first: `head_targets: "rapid,fc"` in the next
  recipe.** Not for independence — see the retraction — but because **every
  unpooled head number in this programme's history is RAPID**: 29 of 230
  archived bundles carry head output and all 29 read `"target": "rapid"`. An
  unpooled FC number would be the first non-RAPID one ever, on a label series
  with 1.7× RAPID's sample count. Price it honestly: `head_targets` is
  recipe-only (the dispatch is AT the 25-input ceiling, so this cannot be
  flipped from the UI), and each extra name costs THREE `probe_head.py`
  processes each paying its own ~30-min anomaly transform — order **+1.5–2 h
  of box wall clock on every run using that recipe**. `probe_kfold`'s
  multi-target loop, by contrast, is free and has been running all along,
  which is why 130 bundles already carry FC/MOVE/OSNAP numbers nobody paid
  for.

**Two measurement traps the audit turned up, recorded so nobody re-finds
them.** (1) `probe_kfold`'s two skip paths are NOT symmetric: an out-of-window
section prints `"<target>: section outside window, skipped"` (:408), but a
MISSING truth key is a bare `continue` (:401) with **no output whatsoever** —
so on the pentad tensor MOVE and OSNAP vanish without a line. (2) The pentad
tensor carries only `rapid` and `fc`: `ml/build_truth_pentad.py:210` is
literally `for name, fn in (("rapid", rapid), ("fc", fc)):`, so MOVE and OSNAP
have no pentad fetcher at all — their sections ARE inside the window, so this
is a labels gap, not a geometry one. Naming `move` in `head_targets` on that
tensor therefore yields three `::warning::` lines and a GREEN run with no
numbers, which is §0.2's failure mode exactly.

---

<a id="e-057"></a>
## E-057 · FGN-mode stage-2: noise-conditioned stencil head trained with fair CRPS — E-057.0 BUILT + CPU-VERIFIED; **E-057.1 pair RE-DISPATCHED after #496's step-6000 OOM (see (h3))**; ensemble roll BUILT; cross-verified against DeepMind's open-source FGN (google-deepmind/weathernext)

TL;DR — can the stencil head stop emitting the conditional mean (the blur
that damps every roll toward climatology) and instead SAMPLE a member of the
predictive distribution, at one forward pass per member, with the noise dose
LEARNED instead of hand-tuned? The FGN move (arXiv:2506.10772): a global
ε ~ N(0,1)^k through zero-init FiLM on every layer's LayerNorm + the fair
CRPS at N=2 as the objective — one change, not two, because under MSE a
noise-conditioned head learns to ignore ε. In the E-052 decomposition this
is the {factorized, ε-sampling} cell; it supersedes E-052.p. Plan with
pre-registered falsifiers:
[E-057](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E057_fgn_head.md);
design argument:
[the FGN addendum](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E052_FGN_addendum.md).

**E-057.0 · implementation + CPU verification of the ε-conditioned head and
the CRPS objective · params: toy configs only (the registered E-057.1 arm is
the monthly xl144 configuration, ~205M + film/eps params) · stage: build /
toy-eval only — NOTHING trains on real data yet · data: synthetic toys ·
arch: `--fgn-eps k` swaps the stock nn.TransformerEncoder for a key-compatible
FiLM container (`_CondLayer`/`_CondEncoder`, adaLN-zero); `--fgn-eps 0`
(default) constructs the LITERAL legacy module tree · steps×batch: toys only,
CPU · resume: n/a.** Spec: `ml/plans/E057_impl_spec.md` (main session);
implementation by an Opus subagent to that spec (§0b); every test re-run and
the full diff read by the main session before commit.

**(a) The flag-off path is verified unchanged at the ARTEFACT level, not
asserted.** A no-flag toy run on the pristine tree vs the patched tree:
`temporal.json` identical in full, every `metrics.jsonl` record identical
(modulo wall-clock), sha256 of every weight tensor in both checkpoints
identical, checkpoint key sets identical (no `eps_gen` key anywhere), stdout
line for line. With `--fgn-eps 0` no module is constructed and no RNG draw is
made, so every archived stage-2 number stays bit-reproducible.

**(b) The zero-init identity holds bitwise, and the one non-identity is
named.** Legacy weights loaded into an `eps_dim=8` twin (strict=False; the
only missing keys are `eps_embed.*`/`*.film.*`): outputs `torch.equal` the
legacy model's in train() and in eval()-with-grad, for ANY ε. In eval() under
`no_grad` the stock layer dispatches to the fused
`torch._transformer_encoder_layer_fwd` and differs by max|Δ| 4.8e-07; with
the fastpath disabled it is `torch.equal` again — the fused kernel is the
whole difference, measured, and no path in temporal.py is affected. So at
step 0 an FGN head IS the deterministic incumbent (the E-057 twin of "r_fore
reads exactly 1.000000 at step 1"), and a legacy checkpoint can warm-start
the trunk (`--init-temporal` wires it, missing-key set checked).

**(c) The torch loss IS the scoreboard.** `fair_crps2` (N=2) and
`fair_crps_ens` (M members, sorted-member identity) pin against
`ml/probscore.crps_ensemble(fair=True)` to 1e-16 on shared arrays; M=1 == MAE
exactly; identical members == MAE exactly. The ε stream is its own seeded CPU
generator (device-independent), saved in every checkpoint beside `torch_rng`
and restored on resume — 6+6 == 12 straight-through, bit-identical over all
tensors INCLUDING the next ε draw; dropping only the ε state diverges, so the
noise stream is load-bearing exactly as the data order is.

**(d) The shared-coin toy reproduces FGN's existence claim in our own
harness.** Law: z_{t+1,p} = z_{t,p} + s_t·pattern_p, ONE coin per time shared
by 32 pixels (conditional mean = persistence; the futures are field-coherent).
A tiny fgn head, trained per-pixel (factorized, zero cross-pixel coupling),
sampled with one ε SHARED across pixels per member: field sign coherence
**0.991** (dev seeds 0–5 worst: 0.952) against **0.149** with independent
per-pixel ε (factorized floor 1/√32 ≈ 0.18), and fair CRPS **39.8% better**
than the same architecture trained on MSE, scored as a degenerate M=1
ensemble. A factorized head with a shared ε emits coherent fields; with
independent ε it cannot.

**(e) A finding worth its own sentence: the fair CRPS does NOT select
coherence — the shared dynamics do.** The toy's first variant gave every
pixel an independent random pattern direction, and the trained head reached
near-optimal MARGINAL CRPS (0.51 vs the 0.50 optimum) with shared-ε coherence
indistinguishable from its independent-ε control (0.10–0.23 vs 0.15–0.20,
three seeds). That is what the objective says it should do: a field-mean
fair CRPS decomposes into per-pixel terms, so nothing in the loss prefers a
member that commits to one coin everywhere. Coherence comes from the
ARCHITECTURE — one global ε modulating one shared trunk, so pixels with
SHARED dynamics respond alike — which the real ocean has and that toy variant
deliberately deleted. Consequence for reading E-057.1 when it runs: the
coherence of real members is an empirical outcome to MEASURE (the pooled-CRPS
instrumentation), never a property the training objective guarantees.

**(f) What is in the diff, and what is refused.** `ml/temporal.py` +502/−30
(all additive, flag-gated): `--fgn-eps` / `--fgn-val-members` (both ride the
`sched:` tail — no workflow edit, the 25-input ceiling is untouched); the
FiLM encoder; fair-CRPS training (two forwards per step on IDENTICAL context,
ε1 ≠ ε2 — any `--input-znoise` corruption is drawn once and shared, so the
CRPS spread term measures only ε); the fixed-bank monitoring ensemble with
NEW record keys `stage2_val_crps` / `stage2_val_member_var` (the ε-collapse
telemetry, F2's live-branch instrument) / `stage2_val_spread_ratio`
(with the (M+1)/M correction); in fgn mode `stage2_val_zmse`/`stage2_amp`
carry the ENSEMBLE MEAN (documented in `stage2_config`, which also records
`stage2_loss_kind: crps2` and `fgn_eval_eps: "zeros"` — the representative
member every pooled legacy point instrument sees, chosen in ONE place,
`fgn_eval_eps()`). Refusals at argv time: fgn × `--direct`, × `--unroll>1`,
× `--unroll-wide` (each feeds back "the prediction", which an ensemble head
does not have — an unmade design decision, not a merge). `forward(eps=None)`
on an FGN head raises — which is the guard that makes `rollout_spatial.py`
REFUSE an E-057 head loudly instead of rolling it clean; the ensemble roll
(M members, ε resampled per step per member — FGN's convention) is its own
follow-up diff and its own experiment entry.

**(g) Cross-verification against the OFFICIAL FGN implementation.** DeepMind
open-sourced FGN as WeatherNext 2
([google-deepmind/weathernext](https://github.com/google-deepmind/weathernext),
Apache-2.0; `weathernext/weathernext2/fgn.py` + `utils/dense.py`), and
E-057.0 was checked against it line by line rather than against the paper's
prose alone. **Matches, exactly:** the CRPS loss — their `crps_loss` at
`unbiased=True` is mean|xᵢ−y| − Σ_{j<i}|xᵢ−xⱼ|/(M(M−1)), which at M=2 is our
`fair_crps2` term for term; the conditional-norm arithmetic — their
`LinearNormConditioning` is `inputs·(1+scale_minus_one)+offset` from a linear
of the conditioning vector, our FiLM exactly; noise as a global vector
entering every block's norms; and the rollout convention — noise is drawn
inside every predictor call, so an AR roll resamples ε each step with the
model seed fixed per trajectory, which is what the E-057 plan registered.
**Three deliberate deviations, each strictly safe:** (i) our film layers are
EXACT-zero-init where theirs are TruncatedNormal(1e-8) — ours makes the
init-identity bitwise instead of 1e-8-close; (ii) we keep the stock
LayerNorm's own affine and modulate on top, where they disable it
(`create_scale=False`) — required for legacy-checkpoint key compatibility,
same function class; (iii) we produce separate (scale, shift) per sub-layer
norm (film → 4·d) where they share one conditioning layer per block — ours
is a superset. **Two of their details imported as registered cautions for
LATER arms, not code changes now:** their loss NaN-masks predictions
wherever targets are NaN so the spread term cannot be inflated at
unobserved points — irrelevant for our dense z-space targets, MANDATORY if
E-057 ever trains a decoded-channel-space CRPS (targets there carry missing
tokens); and their residual-target shift (subtract the per-sample target
deviation) exists for AR-rollout training where targets vary per member —
irrelevant at U=1, mandatory if an fgn head is ever rollout-fine-tuned.

<a id="e-057-1"></a>
**(h) E-057.1 DISPATCHED 2026-08-27 ~17:5xZ — the two-seed pair.**

**#492 (E-057.1a FGN pair seed 0) · eps~N(0,1)^32 through zero-init
conditional LayerNorms + fair CRPS at N=2 replaces MSE, input-znoise OFF ·
params ~211M-class (xl144 + eps path; read the true count off
stage2_config.params_M) · stage stage-2 · data family3_na025 · arch temporal
1024×16 K24 stencil 145 sunflower-144 (the E-036 geometry verbatim) ·
steps×batch 200,000×256 (~2× step cost — two forwards per step) · resume
run-62,run-63 (frozen f3 anchor codec).** On `gpu-box-40623952`
(online-idle at dispatch → starts immediately). **#493 (E-057.1b, seed 1)**:
identical but `seed:1`, queued on `gpu-box-32966687` BEHIND #490/#491 —
the E-056 queue is untouched and ahead of it. Dispatch blocks are the
verbatim #359/#360 INPUTS_JSON (recovered from their logs) with exactly
four changes, each recorded: `seed`, `--input-znoise 0.7 → 0` plus
`--fgn-eps 32 --fgn-val-members 8` on the sched: tail, `head_probe
false → "true"` (the §3 unpooled ruling postdates E-036; copying the old
default verbatim would repeat the #414–#419 missing-head-number failure),
and `job_timeout 1800 → 2100` (~2× #359's measured 15.9 h + margin).
Plans: `ml-metrics/plan-492.json` / `plan-493.json` (expdecay, peak 1e-3,
cooldown 0, milestones 600/60k/120k — plan-359's values).

Hypothesis and controls (pre-registered,
[the plan](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E057_fgn_head.md)):
the LEARNED perturbation + proper score replaces the hand-dosed input noise
— ensemble-mean corridor AUC read against the znoise pair (**0.7235/0.7240**)
and the clean pair (0.6781), both two-seed controls at this exact geometry.
**F1**: below the znoise band ⇒ the FGN move fails to replace znoise on the
incumbent scoreboard. **F2**: `stage2_val_member_var` sliding to 0 ⇒ ε
ignored (collapse); it is on the live branch precisely so this is visible
DURING the run. **F3** (at roll time): the dispersion battery before any
rolled number is quoted as forecast. The corridor read needs the ensemble
roll (E-057 roll diff — in progress as this is written); the roll is scored
AFTER training from the published heads, so the training pair does not wait
on it. **Registered hazards:** (i) at ~2× of 15.9 h both runs likely exceed
the 24 h job-token expiry — the archive steps will 401-and-report-success
(§8, the #419 mechanism), so **both runs are hand-harvest jobs**: pull
`run-49{2,3}.jsonl` + `temporal*.pt` from the box/live branch when they
finish, never trust a green archive step. (ii) The FGN-mode
`stage2_val_zmse` is the ENSEMBLE-MEAN MSE (M=8 fixed bank) — comparable in
spirit but not in mechanism to the deterministic curves; the config record
says so. (iii) Cost: ~2 × 30 h-class box-time, ≈$15–20 total at $0.24/h —
recorded at dispatch per §3's cost rule.

**(h2) The dispatch afternoon, as it actually went — four runs, two sick
boxes, one queue-priority call.** All caught by first-minutes verification
(§2), none by luck. **#492 (E-057.1a seed 0, first attempt)** died at the
GPU check on `gpu-box-40623952` — "GPU present but no torch wheel can reach
it", the dead-GPU-handle signature; the host then dropped offline. Re-issued
as **#496 (E-057.1a seed 0, re-dispatch)** on the restarted
`gpu-box-42005419`, which passed every check and is TRAINING (config
therefore validated end-to-end). Meanwhile the E-056 session's #490/#491 had
failed on their own and were re-dispatched as #494/#495 — which are
BOX-LOCAL to `gpu-box-32966687` (run-485.pt) — while my #493 (seed 1) held
that box for a 30 h run. **#493 was cancelled at ~1 h by this session to
hand the box back**: two short gate runs that can run nowhere else outrank
one long run that can run anywhere, and E-056 was queued first. #494 picked
the box up (E-056 restored, #495 behind it). Seed 1 then took two more
attempts: **#497** died before setup on `gpu-box-30257785`'s FULL DISK, and
**#498** was a deliberate ~4-minute probe of the recovered
`gpu-box-40623952`, which reproduced the identical GPU failure — the box is
a lemon, not a transient. Both sick boxes are STOPPED (no idle burn;
40623952 = Vast 48478310 is a destroy-candidate, 30257785 = Vast 47726876
needs disk triage). **#499 (E-057.1b seed 1, fourth dispatch)** is queued on
`gpu-box-32966687` behind #494/#495. Wasted spend across the three failures:
≈$1–2 (all died in minutes). Fleet now: #496 training (gpu-box-42005419),
#494 training + #495 + #499 queued (gpu-box-32966687).

**(h3) #496 (E-057.1a seed 0, re-dispatch of #492) DIED at ~step 6000 —
OOM in the M=8 monitor validation — and went GREEN anyway (the §7
green-but-void signature, on a healthy box this time).** The run showed
"completed success" after ~100 minutes of a ~27 h job; the tell was the
archive: `run-496.jsonl` ends at `stage2_step` 4000 and `probes-496.json`
lists **no `temporal.json`** — temporal.py is backgrounded behind
best-effort guards, so its death does not colour the run, and the K-sweep
probes ran on and archived normally. The log has the mechanism exactly:
the FGN monitor block pushed the FULL 4096-window monitor batch through the
head once per eval member (M=8), and at stencil 145 the input
`torch.cat` at temporal.py:287 is 4096×24×~9292 fp32 ≈ **3.65 GB per
forward** — the OOM's 3,651,141,632 bytes to the byte. Interleaved with the
two-forward CRPS training steps this fragmented the 24 GB card (13.41 GiB
reserved-but-unallocated at death); the step-2000 val already OOM-retried
(18:22:30Z warning), the step-6000 val died (18:55:04Z,
`torch.OutOfMemoryError: tried to allocate 3.40 GiB`). Same failure class
as E-027 #285/#286, which is why `_chunked_forward` exists — the new FGN
monitor loop just didn't use it. **Fix (this commit): the FGN monitor eval
is chunked to 512-window slices per member** (peak input cat 3.65 GB →
0.46 GB; concatenation over disjoint slices is exact for a row-wise
forward; the legacy deterministic branch is untouched, so archived-path
purity holds by construction; `tests/test_fgn_head.py` +
`tests/test_fgn_roll.py` green, toy batches < 512 ⇒ bitwise-identical).
Two side-findings from the same log: the ~30-min head snapshot upload
422'd (`run-496-temporal-latest.pt` "the head is only on this box" — a
partial/conflicting release asset; snapshot_head.sh should delete-then-
upload, open follow-up), so nothing of #496 survives — nothing worth
rescuing at 4,000/200,000 steps; and the healthy first-minutes readings
(crps2 config, LR 9.66e-4 on plan, member_var 0.871 falling from 1.394 —
ε NOT collapsed, F2 negative so far) validate the config itself. #499
(E-057.1b seed 1, queued) was pinned to the pre-fix sha and would have died
identically at its own step-6000 val — cancelled and re-dispatched behind
#494/#495. Re-dispatches, both on the fixed sha (14bb379): **#500
(E-057.1a seed 0, third dispatch) TRAINING on gpu-box-42005419** (verified
in_progress on the right runner within a minute) and **#501 (E-057.1b
seed 1, fifth dispatch) queued on gpu-box-32966687 behind #494/#495**
(E-056's box-local pair). Cost of the OOM: ~1.2 h of one box, ≈$0.35.

**(i) The ENSEMBLE ROLL is BUILT and CPU-VERIFIED** — `ml/rollout_spatial.py`
+779/−13, spec `ml/plans/E057_roll_spec.md`, tests
`tests/test_fgn_roll.py` (new, 759 lines, 7 checks). A head whose checkpoint
carries `fgn_eps > 0` rolls as **M member trajectories per start**
(`--ens-members`, default 8; per-member CPU generators at
`ens_seed·1000003+59+m`, continuing temporal.py's +57/+58 family; ONE ε per
(member, step) shared across all pixels — FGN's convention, verified against
weathernext). The ENSEMBLE-MEAN field goes through the UNCHANGED
`accumulate`/`skill_block`/audit lines — F1's corridor comparison against
the znoise pair is mechanical, and `meta.fgn` marks the entry. Beside it,
NEW keys only: per-scope `ens_prob` (fair CRPS / spread–error with (M+1)/M /
the mse_mean+mean_var=mse_sample decomposition, all via ml/probscore, under
the weathernext NaN rule), `amoc_bands_ens`(+`_unpooled`) with a recorded
dip-Brier threshold, and the **dispersion battery** on the long/future
rolls (`long_dispersion`/`future_dispersion`: per-step member spread of the
transport read + corridor field variance, WITH `field_var_floor` — the
float32 one-pass-variance cancellation bound, derived not guessed — so a
collapsed head reads as "at the floor", never as a small number). Member 0
only is dumped. **Deterministic-path purity is the acceptance bar and is
tested three ways**: `roll_step(eps=None)` bitwise vs an inline pre-patch
copy AND the pristine module (6 stencil×chunk cases); the whole evaluator
deep-equal (108 keys) against the pristine tree on a deterministic toy;
zero E-057 keys and provably inert `--ens-members` flags for deterministic
heads. Two defects found in review and fixed: the float32 variance floor
above, and `%.6g` serialisation (fixed-decimal rounding wrote a collapsed
head's 1e-8 member variance as 0.0). One measured deviation from the spec:
bitwise chunk-invariance does not exist even on the DETERMINISTIC path
(5.96e-08 float-dispatch floor), so the shared-ε test asserts the exact
property instead — every chunk of a member-step received the identical ε —
plus ε adding nothing beyond that floor. All rollout CPU suites green
except three that diff against pinned historic shas (impossible in a
shallow sandbox checkout, verified pre-broken on the pristine tree) and two
verified pre-existing failures unrelated to this diff.

**(j) Fleet note by the E-053/E-054/E-056 session, 2026-08-27 ~20:40Z —
seed 1 moved off the shared box as #502, pair now parallel.** #501
(E-057.1b seed 1, fifth dispatch) was queued on `gpu-box-32966687` behind
#494/#495 — ~17 h of E-056 work before it could even start its own ~27 h.
Under Chris's parallelization directive ("Any additional boxes and
parallelization is welcome"), #501 (queued, nothing spent) was cancelled
and re-dispatched as **#502 (E-057.1b seed 1, identical config: #496's
INPUTS_JSON with `seed:1`, ref main ⊇ the 14bb379 OOM fix — same sha
class as #500)** pinned to the freshly rented `gpu-box-46292015` (Vast
48937793). Both seeds now train in parallel; the pair lands ~24Z 08-28
instead of ~2 days out. CAVEAT on the record: the pair is now CROSS-BOX —
tensor and Z are release-pulled so the measured box-effect cause is
removed, but the pair spread conflates box with seed (§3b's cross-box
caution); read the pair with that in mind.

**Verification: `tests/test_fgn_head.py` (new, 8 suites) green; existing
`test_resume_temporal` / `test_direct_heads` / `test_e022_stencil` (24
pytest cases) green; `test_probscore`, `test_e044c_knobs`, `test_e029_znoise`
green.** (Three unrelated suites that diff temporal.py against pinned
HISTORIC shas cannot run in a depth-1 sandbox checkout and were verified
pre-broken on the pristine tree.) Cost: $0 GPU, ~45 min CPU. **Next:
E-057.1 (the two-seed monthly xl144 pair, znoise OFF, controls clean 0.6781 /
znoise 0.7235) is the next GPU dispatch after the #489–#491 queue clears —
its §0d entry is written at dispatch, per the plan's falsifiers F1–F3.**

<a id="e-053"></a>
## E-053 · The space-time stencil — RESOLVED 2026-08-27: all four E-053.1 arms in, wave synthesized (frames, not placement, are the binding resource) (Chris's direction: "take the sunflower to the 4th dimension")

TL;DR — today's stencil samples a dense K×145 slab of spacetime (every slot at
every frame, attention over time only); E-045 says span is the axis and the
seasonal analog is its registered mechanism, so most of the slab is plausibly
redundant. E-053 gives each slot its own (Δx, Δy, Δt) — the sunflower
generalized to spacetime under an advective metric — to decouple span from K.

**E-053 · plan only · params n/a (arms would run #427's exact 206.5M stage-2
configuration) · stage: design + a $0 CPU measurement · data
family4_na025_pentad_r2 via the published Z · arch unchanged for stage .1
(non-uniform frame times + Δt encoding), point-cloud attention deferred to a
gated stage .2 · steps×batch: 20k×256 per arm when dispatched · resume
run-415 (frozen).** Staging: **E-053.0** measures the advective cone c from
the published Z's space-time cross-correlation (never hand-pick it);
**E-053.1** is three ~$3–5 arms on the existing trainer — A1 analog-only
(K_eff 16, 2-y span: is span's value the analog?), A2 log-ramp (K_eff 24 ≈
K=144's 0.0820 at 1/6 the frames?), A3 decade (K_eff 32, first 10-y span
ever); controls are the measured 0.5056/0.5045 (K=24) and 0.0820 (K=144).
**E-053.2** (true spacetime point cloud, unifies with the field-head line and
later with a depth/height coordinate) is gated on .1. **Registered danger:
analog pins are a lookup mechanism by construction — the E-043b-PHASE replay
battery is mandatory before any E-053 roll number is quoted.** Full design,
falsifiers and cost arithmetic:
[the E-053 plan](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E053_spacetime_stencil.md)
· [the two-slide deck](https://blauewelt.github.io/earth/ml/figures/spacetime_stencil.html).

**E-053.0 MEASURED, 2026-08-26 ~15:30Z ($0 GPU, `ml/spacetime_corr.py`): THE
CONE IS REFUTED AT ITS OWN INSTRUMENT — THE BALL FORM WINS — AND THE ANALOG
IS VISIBLE IN THE SUBSTRATE.** Over 7,308 deseasonalized pixel pairs
(55–4,400 km, ±180-day lags, ordering asserted against the 86,698-pixel
inventory), the cross-correlation ridge sits at **lag 0 at every
separation's median** (p75 moves only in the far tail: 5 d at ~2,000 km,
30 d at ~4,400 km) — no single advective c exists at this instrument, so
each slot's Δt is placed by the agnostic ball/ramp, not an advective cone
(Chris's "equally distributes also in the time dimension", measured right).
And the centres' own deseasonalized autocorrelation falls to **0.100 at
180 d then RISES to 0.143 at 1 y and 0.113 at 2 y** — the seasonal analog
carries real beyond-climatology information, directly motivating A1's pins.
Numbers + caveats (argmax-lag blindness to sub-dominant advection; wind
channels inflate lag-0): plan §4 and `ml/runs/e053_cone.json`.

**E-053.1 DISPATCHED, 2026-08-26 ~22:09Z — three arms, sequential on
gpu-box-32966687 (Vast 48520137; the HK box 47724565 answered
`resources_unavailable` with a queued state change — the idle-burn risk is
logged in `claude/expectations.md`).** All three: **#427's exact configuration (206.5M-class
xl144+znoise head, stencil 145, ring spiral, znoise 0.7, grad-clip 128,
seed 0) · stage stage-2 · data family4_na025_pentad_r2 · resume run-415
(steps 197,428 — read off the artefact, nothing trains in stage 1) ·
temporal_steps 20,000 × 256 · recipe xl144-zn-pentad-nolonhold + the
`--frame-offsets` tail (instrument commit 5734328, bit-identical off).**
Params shift only with the pos table: 206,527,520 (K_eff 16) / 206,535,712
(24) / 206,543,904 (32). Controls, already measured: uniform K=24 (115-d
span) 0.5056/0.5045; uniform K=144 (715-d span) 0.0820 (#478). Read-out:
the one-step z ratio vs each arm's own persistence (unchanged target ⇒
unchanged denominator); pooled probes emitted, excluded. n=1 directions
(§3b). NO ROLL from these heads without the E-043b-PHASE battery — and
rollout_spatial refuses offset heads outright until offset-aware assembly
exists.

| run | arm | K_eff · span | offsets | registered reading |
|---|---|---|---|---|
| #486 | **A1 analog-only** | 16 · 735 d | dense 0…−6, −8,−10,−12, pins −73±1, −146±1 | ratio ≲0.12 ⇒ span's value IS the analog (sparsifies ~10×); ≳0.3 ⇒ dense slow-mode context matters |
| #487 | **A2 log-ramp-24** | 24 · 715 d | log-spaced 0…−143 (incl. −73) | ≈0.08 at 1/6 the frames ⇒ span decoupled from K; ≈0.5 ⇒ the slab was load-bearing |
| #488 | **A3 decade-32** | 32 · 3,650 d | log ramp 0…−730 + pins at −1…−10 y | first >2-y span at any cadence; ratio < 0.0820 ⇒ span still paying (decadal territory); ≈0.082 ⇒ 2 y saturates |
| #489 | **A4 uniform-24** | 24 · 690 d | uniform 0,−6,…,−138 (registered 05:50Z 08-27, after A1+A2) | ≈0.15 ⇒ spacing irrelevant, slab density load-bearing; ≲0.10 ⇒ uniform beats log — spacing artefact in E-045's flatness |

**11:45Z 08-27 — A3's VERDICT: 0.1400 — THE FIRST EVIDENCE THAT CONTEXT
BEYOND TWO YEARS CARRIES INFORMATION.** #488 (A3 decade-32, span 3,650 d)
final: z 2.9761 / 21.2544 ⇒ **one-step ratio 0.14002** (offsets echoed,
integer-year pins −365…−730 present; kfold 0.654 [0.599, 0.703],
indistinguishable from A1/A2's). Against the registered two-sided reading
(< 0.0820 ⇒ decadal territory at any price; ≈0.082 ⇒ 2 y saturates): 0.140
does NOT beat the dense 2-y slab — but within the sparse family, at the
same log-ramp-plus-pins construction, extending the span 2 y → 10 y with
eight extra frames improves the one-step **10.3%** over A2 (0.1561 →
0.1400), making A3 the best sparse arm of the wave. Registered caveat on
the comparison: A3 has more frames (32 vs 24) AND more span — the two are
confounded within this pair; A4 (uniform-24, running next) calibrates the
frame-count axis. Direction (n=1): the decade is not empty — years 2–10
carry real one-step information, which is the first positive result at any
span beyond 730 d in the programme and the motivation to fold long-span
pins into the next full-budget design rather than treat 2 y as a ceiling.

Cost ≈ $1.1–1.5/arm (measured 0.264 s/step × 20k ≈ 97 min stage-2 + probe
tail), job_timeout 600 min each (the E-045 HK lesson). A3's pool shrinks by
construction (t ≥ 730: ~77% of anchors remain) — printed count checked at
first minutes, not a bug. First-minutes checks on #486 all passed:
`frame_offsets` echoed in the config record, K 16, params_M 206.528,
monitor val_persistence **21.44621 — bit-identical to #427's**, the
denominator-invariance the instrument promised.

**01:35Z 08-27 — A1's VERDICT: 0.1858, BETWEEN THE REGISTERED THRESHOLDS —
THE ANALOG IS REAL BUT IT IS NOT THE WHOLE OF SPAN.** #486 (A1 analog-only,
K_eff 16, span 735 d) finished stage-2 and probes; the final eval record:
z_mse_model 3.9495 / z_mse_persistence 21.2544 ⇒ **one-step ratio 0.1858**
(`frame_offsets` echoed in the result's scale block:
`-147,-146,-145,-74,-73,-72,-12,-10,-8,-6,-5,-4,-3,-2,-1,0`). The
pre-registered reading was two-sided — ≲0.12 "span's value IS the analog"
vs ≳0.3 "dense slow-mode context matters" — and 0.1858 lands between, so
neither clean claim is licensed. What IS licensed: (a) 16 frames whose only
reach into the past is a dense recent month plus six analog-pin frames buy
**2.7× over the uniform K=24 head at the same frame budget** (0.5056/0.5045,
115-d span) — the pins carry real span value, ~55% of the full span effect
in log terms; (b) they do NOT reach uniform K=144's 0.0820 (2.27× short),
and E-045's A2a control (24 uniform frames at 30-d stride over the same 720-d
span, ratio 0.0721) already showed distributed coverage of the INTERMEDIATE
times matching the dense slab — so the missing half of the effect plausibly
lives between −12 and −72, exactly where A1 has no frames. **#487 (A2
log-ramp-24, queued behind #486's archive tail) is now the decisive arm:**
log-spaced frames cover those intermediate times at the same K_eff as A2a —
≈0.08 would say span is fully decoupled from K with the analog as one
ingredient among several; ≳0.15 would say the SPACING (log vs uniform)
matters too. Secondary read-outs, not the verdict: rapid_r_kfold 0.647
[0.594, 0.696], deseas 0.548, raw 0.583 — the probe does not distinguish
the arms at n=1. n=1 direction per §3b; NO ROLL without the E-043b-PHASE
battery (the pins are a lookup mechanism by construction).

**05:45Z 08-27 — A2's VERDICT: 0.1561 — LOG SPACING DOES NOT SUBSTITUTE FOR
THE SLAB; SPAN IS NOT DECOUPLED FROM K BY SPARSE SAMPLING.** #487 (A2
log-ramp-24, span 715 d) final: z_mse_model 3.3183 / z_mse_persistence
21.2544 ⇒ **one-step ratio 0.1561** (offsets echoed; params 206,535,712).
Registered reading was ≈0.08 "span decoupled from K" vs ≈0.5 "slab
load-bearing" — 0.1561 lands between again, but the direction is
unambiguous: 24 log-spaced frames over the full span recover **65% of the
span effect in log terms** (A1's pins: 55%) yet remain **1.9× short of
uniform K=144's 0.0820**. Together A1+A2 localize the effect: it is
GENUINELY DISTRIBUTED — a long span helps under any sampling (0.5056 →
0.156–0.186), but the dense slab's remaining 1.9× is carried by the many
intermediate frames sparse patterns skip, and no 16–24-frame skeleton
found it. The 4-D sunflower's premise ("most of the slab is redundant") is
half-right at best at this budget. Secondary: kfold 0.661 [0.612, 0.708]
vs A1's 0.647 — indistinguishable at n=1. **A correction to A1's block
above:** its appeal to E-045-A2a's 0.0721 as "distributed coverage matches
the slab" was cross-instrument — A2a's ratio lives at a 30-day target and
its own 30-day persistence denominator, not comparable to these 5-day-target
ratios. A2 was the like-for-like test, and it says the slab is (partially)
load-bearing. What A2 does NOT separate is spacing from target-step: it
differs from A2a in BOTH. A4 (registered below, dispatched behind #488)
isolates spacing.

**A4 REGISTERED 2026-08-27 ~05:50Z (before dispatch): uniform-24 at
stride 6 — 5-day target.** Offsets `0,−6,−12,…,−138` (24 uniform frames,
span 690 d), everything else bit-identical to A2 (same K_eff 24, same
206,535,712 params, same target t+1 at 5 d, same denominator). Registered
two-sided reading: **≈0.15 (A2-like) ⇒ spacing is irrelevant — sparse K is
the binding limit and the slab's density is load-bearing, full stop;
≲0.10 ⇒ UNIFORM spacing specifically beats log** (log's clustering near
Δt=0 wastes frames on the fast-decaying own-history the head barely
needs) — which would also mean E-045's span-fixed "flatness" was partly a
spacing artefact and cheap uniform-sparse heads reopen. Queued as #489
behind #488 (A3); ~$1.5.

**17:4xZ 08-27 — A4's VERDICT: 0.14116 — SPACING IS IRRELEVANT AT K=24;
THE SLAB'S DENSITY, NOT ITS PATTERN, IS WHAT SPARSE SKELETONS MISS.**
#489 (A4 uniform-24, span 690 d) final: z_mse_model 3.0002 /
z_mse_persistence 21.2544 ⇒ **one-step ratio 0.14116** (uniform offsets
0,−6,…,−138 echoed in the config; kfold 0.663 [0.61, 0.712]). The
registered two-sided reading resolves cleanly to the first branch:
0.141 is A2-like (0.156), nowhere near ≲0.10 — uniform does not beat
log by any margin that matters (Δ 0.015 at n=1, and §3b has no measured
pair at this tier). Note #489 carries NO unpooled keys — it was
dispatched 05:41Z, before E-055 landed at 12:15Z; refs pin at dispatch.

**THE FOUR-ARM WAVE, SYNTHESIZED (A1 0.1858 · A2 0.1561 · A3 0.14002 ·
A4 0.14116, against controls K=24-uniform-115d 0.5056 and K=144-dense-2y
0.0820, all n=1 at 20k steps):**

1. **The span effect is DISTRIBUTED, not an analog lookup.** A1's pins
   alone (dense own-history + −1 y/−2 y sunflowers) recover 55% of the
   span effect in log terms; every 24-frame pattern that also covers the
   intermediate times recovers ~65%. The E-045 mechanism story ("span
   buys the seasonal analog") is at best half the effect.
2. **Spacing does not matter at K≈24–32.** Log (0.156), uniform (0.141)
   and decade-log-with-pins (0.140) sit within 0.016 of each other —
   inside anything plausible for this tier's unmeasured noise. What all
   three share is K_eff ≈ 24–32; what separates them from 0.0820 is
   K=144. **Frames, not placement, are the binding resource.**
3. **The dense slab keeps a real 1.7–1.9× edge** over every sparse
   skeleton at 1/6 the frames — the 4-D sunflower's premise ("most of
   the slab is redundant") is refuted at this budget. E-053.2's
   point-cloud architecture loses its cheap-span motivation; it survives
   only as the field-head unification (E-052's FieldDiT), not as a
   stand-alone build.
4. **A3's decade pins are ≈ free span extension**: 10-year span at K=32
   matches 690-day at K=24 (0.140 vs 0.141) — going long costs nothing
   but also *bought* nothing measurable beyond 2 y at this K. The
   beyond-2-y axis stays open only at high K — which is exactly what
   E-054a (400k K=144) and, if it passes, E-056b (K=144 on tokens)
   price out.
5. **Practical consequence for the programme:** the cheap road to more
   span is MORE FRAMES CHEAPER, not cleverer placement — which is the
   token substrate's (E-056) case in one line, and why it was
   prioritized as the efficiency gate before the next full-budget
   pentad head.

---

<a id="e-056"></a>
## E-056 · Tokens as the forecasting substrate — the "true FSQ" gate, DISPATCHED 2026-08-27 ~13:0xZ (Chris: "let's continue with FSQ then")

TL;DR — E-050's warm-start codec compresses each pixel-timestep to a
16-bit alphabet (d_z 6 through FSQ [8,8,8,5,5,5] ≈ 64k codes) — a ~30×
smaller state than d_z 32. This week's span results say long dense history
is what wins, and long dense history is expensive in exactly the currency
tokens are cheap (input volume, gather bandwidth, host pipeline). E-056
gates the token road: do stage-2 heads trained ON the warm-quantized z
forecast competitively — and what does K=144 actually cost on it?

**E-056a · stage-2 head at the standard pentad configuration over the
E-050 warm-FSQ codec's z (d_z 6, quantized) · params ~206.5M-class head
(input proj shrinks: d_z 6) · stage stage-2 · data family4_na025_pentad_r2
· arch head 1024×16 K 24 stencil 145 ring spiral znoise 0.7 grad-clip 128
seed 0 · steps×batch 20k×256 · resume run-485@260k (fsq-warm final;
recipe f4r2-40M-dz6-fsq65k-warm; stage 1 trains nothing).** Registered
readings, controls E-046's lattice-d_z-32 0.4394 and continuous 0.5056
(both K=24 20k): **≲0.44 ⇒ the 16-bit token matches/beats the d_z-32
lattice as a substrate — the token road opens at 5% of the state size;
≳0.50 ⇒ warm quantization survived training but lost the forecastable
information — the audit's question answered from the stage-2 side.**
**E-056b · the SAME over K=144** (`--K 144` window tail): controls #478's
0.0820 and E-051's step time; the run's measured s/step IS the efficiency
datapoint (d_z-6 gather is ~5.3× less data than d_z-32). Registered:
≈0.08 at materially faster steps ⇒ dense long context becomes cheap —
the next full-budget head should train on tokens; ≳0.15 ⇒ the token
substrate degrades under long context and the road needs the audit's
diagnosis first.

Known risks registered at dispatch: (1) **the 260k warm-FSQ final exists
box-locally on 48520137 ONLY** — `model-checkpoints-v1` carries
`rescued-orphan-latest-485.pt` which this session OPENED and found to be
the step-200,000 PRE-FSQ parent (fsq args empty), so the canonical
`run-485__pixelmae.pt` fallback is deliberately left absent (fail-fast
beats silently resuming the wrong codec); the first-minutes check MUST
read the `resumed` record's `at_step: 260000` and the config's
`fsq_levels: 8,8,8,5,5,5`, and the job's own publish step is what makes
the 260k final durable at last (E-050's §5.26 gap, flagged to its
session via expectations). (2) znoise σ=0.7 is an absolute dose calibrated
on d_z-32 z; on the quantized d_z-6 scale the RELATIVE dose differs —
read `input_znoise_rel_pers` in the monitor and record it beside the
verdict (E-045-A4's lesson: noise dose does not transfer as a constant).
(3) The embed pass writes the FIRST token-Z cache (≈3 GB vs 16.24 GB) —
publish it; it is the efficiency artefact. n=1 directions per §3b.
~$1.5 (a) + ~$2–3 (b incl. embed), sequential behind #489 (E-053.1-A4).

**08-28 06:00–07:00Z — BOTH E-056 ARMS DIED, THE GATE IS RE-DISPATCHED, AND
THE E-050 CODEC IS DURABLE AT LAST.** Diagnosis of the overnight state:

- **#494 (E-056a, K=24) was CANCELLED at 04:39:52Z by an actor outside this
  session**, 15 minutes into its stage-2 training (`stage2_step` 2800 of
  20,000, val_zmse 0.3807 ⇒ ratio 0.600 against the token-scale denominator
  0.63451 — falling, but far from a verdict). Everything before that was
  healthy: `resumed at_step 260000`, `fsq_levels 8,8,8,5,5,5`, monitor
  written. No verdict; no archive (the archive step never ran), but the
  live branch survived and carries the 39 records.
- **#495 (E-056b, K=144) is VOID — green with no `temporal.json`** (§7's
  exact signature: `probes-495.json` lists only `probe_sequence.json` and
  `provenance.json`; `run-495.jsonl` has four records and zero
  `stage2_step`). It died of **CUDA OOM on its FIRST forward pass** — the
  feed-forward block asked for 603,979,776 B with 599,523,328 B free on a
  25.25 GB card — `temporal.py:294`, `torch/nn/modules/transformer.py:977`.
  **The mechanism is the box, not the substrate: #478 — the K=144 control
  this arm must be read against — ran the SAME batch 256 at K=144 on a
  LARGER card.** So E-056b needs an 80 GB box (the fleet has two H100s,
  both stopped) to keep batch-256 comparability; halving the batch would
  buy the fit at the price of confounding the very comparison the arm
  exists to make. Registered for the re-dispatch, not improvised now.
- **Two artefacts survived and change the economics.** #495 published the
  **token-Z embed cache durably before dying** —
  `Z_867532fe7b_37e146384b.npy`, 3.04 GiB, 3/3 chunks VERIFIED on
  `embed-cache-v1` — so every future E-056 arm skips the ~95-minute
  rebuild. And #495's orphan-rescue step captured #494's box-local
  checkpoint: `rescued-orphan-latest-495.pt` was downloaded and OPENED
  this morning (§0.15) and reads **step 260000, d_z 6, fsq_levels
  8,8,8,5,5,5, fsq_bound ln, 512x12 patch 1** — it IS E-050's warm-FSQ
  final. **It was therefore published as `run-485__pixelmae.pt`
  (455,676,349 B, 2026-08-28 06:47Z), retiring the two-day-old
  box-local-only risk**: E-050's 260k codec is now durable on the release
  and any box can resume it. (The older `rescued-orphan-latest-485.pt`
  remains the step-200,000 PRE-FSQ parent and must never be published
  under the canonical name — that distinction is why the name was held
  empty until a file could be opened and verified.)
- **RE-DISPATCHED as #504 (E-056a-R, K=24, znoise 0.7)** on
  gpu-box-32966687 at 06:52Z — the box was idle and billing, the codec
  and Z are local, so it started immediately; ~2.2 h. A dose-matched twin
  (**E-056a-CLEAN, `--input-znoise 0.12`**) is registered to run beside it
  the moment a second box comes up: 0.12 = 0.7 x (0.15116 / 0.87878),
  which puts the token-scale relative dose where 0.7 put it on d_z-32.
  Running the pair together makes the token verdict readable in ONE wave
  rather than a round trip, which is the whole reason the confound was
  pre-registered rather than discovered.

**BOTH REMAINING ARMS DISPATCHED 08:40Z, and one of them died in `Set up
job`.**

- **#505 (E-056a-CLEAN, K=24, znoise 0.12)** — FAILED before a single step,
  and it is worth recording because the failure is a fleet mode rather than
  a science one: `##[error]No space left on device` while unpacking
  `actions/upload-artifact` into `/opt/runner/_work/_actions/`, on
  gpu-box-30257785 (Vast 47726876, a 100 GB box) 4 seconds after the runner
  picked the job up. The job's own hygiene step never runs, because the
  failure is upstream of step 1 — so a full-disk box cannot clean itself and
  is `disktriage` material, not a host. Box stopped immediately (idle burn,
  §7). Re-dispatch goes on the slot #504 frees, which also carries the token
  Z warm. Cost: ~4 minutes of a $0.321/h box, no GPU.
- **#506 (E-056b, K=144, znoise 0.7)** — dispatched to **gpu-box-40024079
  (Vast 48632885, H100 SXM 80 GB, $2.028/h)**, and the CARD is the change,
  deliberately. #495 measured that K=144 at batch 256 does not fit a 24 GB
  4090; its control #478 ran that batch on a bigger card, so halving the
  batch would have confounded K with batch on the one comparison this arm
  exists to make. It reached the `Train` step cleanly. The codec comes from
  the release now, not box-local, which is what made a fresh 80 GB host
  possible at all.
  **Its dose is 0.7, and that is on purpose.** The pre-registered token-scale
  confound makes the token-vs-d_z-32 LEVEL uninterpretable at 0.7 — but it
  does not touch this arm's question: #504 and #506 share the dose exactly,
  so K=24 vs K=144 WITHIN tokens is a clean within-dose contrast, the same
  shape as #478 against its K=24 twin on d_z-32. The dose-matched LEVEL comes
  from #505's re-dispatch. Read the three together, never any one alone.

**#506 CANCELLED 12:30Z — THE H100 NEVER USED ITS GPU. ~$8 AND NO NUMBERS.**
The arm reached `stage2_config` and `stage2_monitor` at 11:39Z with everything
correct (`params_M 202.745 · K 144 · d_z 6 · batch 256 · stencil 145 ·
val_persistence 0.63451` — the token-scale denominator, so it WOULD have been
directly comparable to #504 — and `input_znoise_rel_pers 0.87878`), and then
wrote **zero** `stage2_step` records in the following 50 minutes.

**The verdict rests on a CONTROL, not on a threshold.** #478 is this arm's own
K=144 control at the identical geometry (1024x16, batch 256, K 144) and its
archived `run-478.jsonl` writes its FIRST step record at **`wall_s 240.3`** —
four minutes. #506 sat at fifty. In the same window its sibling #507, the same
pipeline over the same codec on a *cheaper* 4090, was running at **0.306
s/step with `gpu_util` 99.99%**. And across the job's whole 3 h 55 m — samples
at 09:56, 10:00, 10:23, 11:40, 11:42, 11:49, 11:59, 12:11, 12:19, 12:29Z —
**every single `gpu_util` reading was 0% while `cpu_util` sat at 95–97%**, with
`fleet_health`'s TELEMETRY (dead-frame) check never firing, so those were real
readings and not the known Vast all-zero bug. The mechanism that fits every
one of those observations: **torch on that container could not see the card**,
which also explains the 2 h 45 m "embed" — a CPU embed — that preceded it.

**The decision rule was PRE-REGISTERED before the evidence closed**, and that
is the part worth keeping: at 12:05Z, with the case still open, the rule
written into `ml/OVERVIEW.md` was *"if a record appears by ~12:30Z at <=3
s/step it runs on (input-starved at worst); if not, cancel rather than spend a
10 h timeout at $2.028/h"*. It was held to at 12:29:35Z rather than improvised
either way. Cancel accepted 12:30:32Z, job `cancelled` 12:36Z, box stopped
12:37Z. **Cost: 3 h 55 m x $2.028/h = ~$7.95**, against the ~$20 the remaining
timeout would have spent. Nothing was lost besides the money — the token-Z
cache was already durable from #495, so no artefact died with it.

**E-056b is HELD, not re-dispatched, and the sequencing is the reason rather
than the boxes.** The other 80 GB host (48495564, $1.567/h) answers
`resources_unavailable` and the default market search returns only 24 GB
cards, so a re-dispatch means renting a fresh 80 GB box — but the better
argument is that **E-056b's question only matters if tokens are viable at
all**. It asks whether the dense K=144 slab helps *within* the token
substrate; #507 is 2-3 h from saying whether the substrate survives at the
dose-matched noise. Spending ~$20 and another 3 h of setup on the K contrast
before that answer is exactly the sequencing §4.4 warns about. Re-dispatch it
if #507 rehabilitates the substrate; drop it if #507 confirms #504.

**#504's TRAINING CURVE IS COMPLETE, 09:42Z — final one-step ratio
0.53873** (val_zmse 0.34183 / the token-scale denominator 0.63451) at step
20,000, having come down 0.600@2800 · 0.574@6000 · 0.556@10000 · 0.548@16000
· 0.537@18000 · **0.539@20000** — a curve that stopped improving around step
16k. Against the registered controls that is **worse than the continuous
d_z-32 arm (0.5056) and far from the lattice d_z-32 bar (0.4394)**, i.e. on
the wrong side of the pre-registered ">=0.50 means quantization lost the
forecastable signal" line. **It is not yet a verdict**, for the reason
registered before dispatch: at znoise 0.7 this arm carries `rel_pers 0.879`
on the token scale where the d_z-32 controls carried 0.151 — ~5.8x the
intended relative dose. #507 is the arm that decides it. (The probe ladder's
tail was still running at 10:00Z; the head numbers land with it.)

**#504 COMPLETE 11:4xZ — THE FULL BUNDLE, AND THE PROBES ARE NOT SUBJECT TO
THE DOSE CONFOUND.** `probes-504.json` carries all eight files including
`temporal.json` (so this is not a §7 green-but-void run) and all three head
probes. Read from the artefact, self-consistently:

| instrument | #504 (tokens, d_z 6) | its own matched control | control's control |
|---|---|---|---|
| one-step `z_t+1` ratio | `0.33292 / 0.62375` = **0.53373** | continuous d_z-32 **0.5056** | lattice d_z-32 **0.4394** |
| unpooled head, RAPID (`probe_head`) | **0.588** [0.534, 0.641] | raw-3x3 **0.693** [0.633, 0.746] | unpooled wind bar **0.690** [0.620, 0.751] |
| pooled ridge on the codec (`probe_kfold.rapid`, legacy) | 0.553 [0.498, 0.606] | pooled wind bar 0.670 | — |
| pooled ridge, Florida Current (`probe_kfold.fc`, legacy, free) | **0.051** [-0.015, 0.122], n 2,490 | pooled wind bar 0.199 | — |
| stage-2 head, pooled (`rapid_probe_kfold`, legacy) | 0.633 [0.582, 0.686] | — | — |
| stage-2 head, UNPOOLED (`rapid_probe_kfold_unpooled`) | 0.562 [0.500, 0.624] | — | — |

Note the ratio's denominator: `temporal.json`'s own final-eval persistence is
**0.62375**, where the in-training `stage2_monitor` read 0.63451 on its
4,096-window draw. The two differ by 1.7%; the table quotes the ratio computed
**within one file**, which is the only self-consistent way to take it. Against
the monitor denominator the same numerator gives 0.5247 — the verdict does not
turn on the choice.

**The part that matters, and it is a distinction nobody had drawn: the
znoise confound does NOT reach the probe rows.** `--input-znoise` is a
STAGE-2 training knob; `probe_head` and `probe_kfold` score the **frozen
codec's** embeddings and never see it. So the forecast row is confounded (5.8x
the intended relative dose, #507 decides it) and **the probe rows are clean**.
They say the 16-bit token embedding carries **less** RAPID-relevant signal than
its own raw-3x3 control (-0.105) and less than wind stress alone (-0.102), and
essentially nothing about the Florida Current (0.051, a CI containing zero,
against a 0.199 wind bar) — on E-001's framing, a codec that subtracts.

**How much weight that carries: a direction, not a level, and §3b is
unambiguous about why.** The head k-fold's measured seed regime is
**0.036-0.245**, so a -0.105 gap sits INSIDE it, and *"no probe number is
readable at n = 1, at any scale this programme has run"*. Every pentad arm is
n = 1. What the rows do have going for them is that they are a **matched
within-run comparison** — all three are the same unpooled attention head, the
same 266-pixel section, the same folds, in the same job, which is exactly the
both-sides-switch-together discipline §3 demands. So: two independent
instruments point the same way, and only one of them is confounded. That is
worth saying, and it is not yet a verdict.

**#507 (E-056a-CLEAN) RE-DISPATCHED 10:19Z, QUEUED BEHIND #504 ON THE SAME
BOX** — deliberately, and it is worth saying why rather than taking the next
free host. #505 died on a full disk; two other hosts were tried in the gap
(47913006 and 48524831 returned `resources_unavailable, state change queued`;
47724559 accepted the start, reported `cur_state: running`, and never
registered its runner across 18 minutes, so it was stopped). Queuing on
gpu-box-32966687 instead means #504 and #507 run **the same configuration on
the same box with the same warm token-Z cache, differing in the dose and in
nothing else** — which is what a dose-matched arm is for, and is strictly
better than the cross-box pair the free-host route would have produced (the
box effect, §7, is 0.041 on the head k-fold). It starts the moment #504
releases the runner.

**#504 as of 08:47Z, step 12,000/20,000: ratio 0.5555** (val_zmse
0.35245 / the token-scale denominator 0.63451), from 0.600 at 2,800 and 0.574
at 6,000 — descending but flattening, and currently WORSE than the continuous
d_z-32 control (0.5056), let alone the 0.4394 lattice bar. Taken alone that
reads "quantization lost the forecastable signal". It must not be taken alone:
at znoise 0.7 this arm carries ~5.8x the intended relative dose, which is the
registered confound. The reading is a direction awaiting #505.

**The prioritization rule this wave instantiates (Chris: "think of the
best way to prioritize such efficiency-inducing experiments"):** an
efficiency experiment is scheduled BEFORE the next big spend on the axis
it cheapens, and never blocks spends on other resource pools. Concretely:
E-056 gates the next full-budget pentad head (a ~$100-class spend that
would train on tokens if E-056 passes) but does NOT gate E-054a/b (TPU
pool, already priced) or the E-051 roll (eval, different question). The
general form: price the gate (~$5–8 here) against (expected saving ×
planned spend on that axis) — tokens at ~30× input reduction against a
$100+/run axis clears it by an order of magnitude; run such gates on idle
box time the moment they clear the bar.

**17:2xZ 08-27 — #490 (E-056a) AND #491 (E-056b) BOTH FAILED in ~11 min
each, and the failure is GOOD NEWS twice over.** The Train step died on
train.py's warm-start guard: *"--fsq-warmstart: the checkpoint ALREADY
carries --fsq-levels '8,8,8,5,5,5', so this is not a warm start"* — the
dispatch reused recipe `f4r2-40M-dz6-fsq65k-warm`, whose
`fsq_warmstart: true` is correct for CREATING E-050 from the continuous
parent and wrong for RESUMING its product. The guard behaved exactly as
designed (FSQ→FSQ with the flag would reinterpret the lattice). The two
findings that came free: **(1) risk (1) above is RETIRED — the box-local
`/opt/earth-cache/ckpt/run-485.pt` on gpu-box-32966687 (= Vast 48520137)
EXISTS and carries the FSQ lattice** (the guard read its args), so the
runner names in my registration and the "48520137" in E-050's notes are
the same physical box; the fail-fast never had to fire. (2) The refusal
came at minute ~11, after the anomaly transform — a §0.16 note for the
workflow: this guard depends only on checkpoint args + flags and could
run before the tensor work. **RE-DISPATCHED 2026-08-27 17:44Z as #494
(E-056a, K=24) and #495 (E-056b, K=144)** with the base recipe
`f4r2-40M-dz6-fsq65k` — verified field-identical minus `fsq_warmstart` —
which is precisely the guard's own prescribed path ("resume it without
--fsq-warmstart to continue the lattice it has"). Queued behind #493
(E-057.1b, another session's FGN seed-1 head) on the same box.
First-minutes checks unchanged: `resumed at_step 260000`,
`fsq_levels 8,8,8,5,5,5`, and `input_znoise_rel_pers` recorded. Cost of
the false start: ~$0.25 of box time, two run numbers.

---

<a id="e-055"></a>
## E-055 · The unpooled stage-2 transport read-out — BUILD STARTED 2026-08-27 (Chris: "let's take 4 first")

TL;DR — geostrophic transport at 26.5°N is the east-minus-west contrast
across the section, and a spatial mean annihilates exactly that contrast;
the 2026-08-21 standing rule ("unpooled is the verdict, everywhere")
converted every codec-side probe, but the STAGE-2 transport read-out —
`hid[:, -1].mean(0)` at ml/temporal.py:2349 and :2181 — was deliberately
deferred (§3/§8: no unpooled counterpart existed, 98 archived runs read the
pooled key, and the roll evaluator's pooled `read_sv` feeds the frozen
gate's pinned reference). This experiment is that deferred change, done as
its own experiment, never silently.

**E-055 · add learned-pool (probe_head-mechanism) transport read-outs as
NEW KEYS beside the pooled legacy ones · params: a tiny attention-pool head
(probe-scale, not a model change) · stage: instrumentation · touches
ml/temporal.py (probe tail: `rapid_r_kfold_unpooled`,
`rapid_r_deseas_unpooled` beside the old keys) and ml/rollout_spatial.py
(an ADDITIONAL `read_sv_unpooled` writing `amoc_bands_unpooled` +
unpooled sv series beside `amoc_bands` — `read_sv`, `GATE_REF`, the
collapse guard and the JAX certificate are UNTOUCHED, per §3's three
exceptions) · no new workflow inputs (the 25-input ceiling) · tests: the
archived-roll identity test must still pass bit-identically, new keys
finite + seed-reproducible on a fixture.** Registered readings: the paired
pooled-vs-unpooled contrast on the SAME predicted states is the number
(§8's never-run paired test, now on stage-2); expectation from the codec
side is a +0.03-class probe gain, and §3b prices probe noise at 0.036–0.245
— so the contrast is read paired, never as two levels. Implementation by
an Opus subagent to this spec; verified by the session before commit.

---

<a id="e-054"></a>
## E-054 · Budget and capacity at the pentad frontier — E-054a (continue E-051) DISPATCHED 2026-08-27 ~10:0xZ at Chris's direction ("let's continue E-51")

TL;DR — E-051's val curve is a clean power law to 200k with no saturation
(0.0812@20k → 0.0414@100k → 0.0330@200k, α≈0.33 in the last doubling) and
its LR had decayed to 1/32 of peak — so the flattening is partly the
schedule, and the cheapest test of "does budget keep paying" is to continue
the run. Fig.~fig_pentad_budget in the paper is the picture.

**E-054a · CONTINUE the full-budget K=144 pentad head from its own 200k
checkpoint to 400k, with a re-armed LR (4e-4, halflife 100k ⇒ lr(200k)=1e-4
≈3× the old terminal rate, lr(400k)=2.5e-5) · params 206.66M (unchanged) ·
stage stage-2 · data family4_na025_pentad_r2 via the published Z · arch
1024×16, K 144, stencil 145 ring spiral, znoise 0.7, grad-clip 128, seed 0 ·
steps×batch: +200k × 256 (400k total) · resume
runs/e051-k144-full/temporal_e051_jax.npz@200k (2.37 GB, WITH optimizer
state — verified on the bucket before dispatch).** TPU v5litepod-4 **SPOT**
(the on-demand quota is held by the relaunched E-052.1 node; preemption is
covered by ckpt_every 2000 + 10-min shipping, resume-by-node-name).
Registered two-sided reading at 400k, vs 0.0330: **≈0.026 (the power law's
own extrapolation) ⇒ budget still paying at K=144 — queue the capacity rung
(E-054b, ~400M) next; ≈0.033 flat ⇒ 200k was effectively converged and
capacity, not steps, is the open axis.** Confound stated at registration:
any continuation re-arms the schedule, so "more steps" and "fresh LR" are
not separable here — a flat result convicts both at once, a falling result
does not attribute between them. Expect a brief val REGRESSION just after
202k (warm-restart transient) — a first-minutes check that panics on it is
misreading the design. Cost ≈ 17 h wall un-preempted; spot ≈ $30–41 at the
day's rate ($82 if it had gone on-demand). NO ROLL from the result without
the E-043b-PHASE battery.

**E-054b · the capacity rung (registered, NOT dispatched):** same recipe
fresh at ~400M (1280×20, K 144, 200k) — ≈32 h train, ≈$155 on-demand /
≈$55–75 spot; HBM fit at 1280 wide × K=144 unverified (E-051 fit 1024×16
with ~2 GB headroom at chunk 256 — first-minutes OOM check mandatory).
Waits on E-054a's reading per the two-sided registration above.

**E-054b · LAUNCH DECISION 2026-08-27 ~20:55Z (Chris: "If more TPU
resources ... speed things up, let's start them"):** the wait-on-E-054a
gate is DROPPED — both branches of E-054a's two-sided registration point
at the capacity rung (a falling curve queues it; a flat one names capacity
the open axis), so no reading can cancel it and holding it bought nothing.
Config frozen in /tmp/e054b_startup.sh (fresh 1280×20, K 144, 200k×256,
LR 4e-4 halflife 100k, znoise 0.7, grad-clip 128, seed 0, run-415 codec +
published Z, node/prefix `e054b-400m`, TAG e054b). The launch itself is
HELD ~2 h by supply, not by choice: the full 7-zone spot ladder returned
nothing at 20:44–20:51Z — us-west1-c and us-west4-a quota-full with OUR
OWN spot nodes (the 4-core/zone preemptible grant = one v5litepod-4 per
zone), everything else capacity-dry (SPOT_LEDGER 08-27 20:44Z). Paying
the ≈$100 on-demand premium to start a 32 h run 2 h early fails the
programme's own arithmetic, so E-054b launches at the 22:35Z wake on the
us-west1-c spot quota E-054a frees at ~22:30Z — with on-demand as that
wake's EXPLICIT fallback if the fresh spot draw refuses (capacity there
is proven only for the node we already hold). First-minutes checks:
the HBM/OOM verdict at 1280×20×K144 chunk 256 (registered risk), LR
4e-4 nonzero, fresh start at step 0 (an accidental resume from a stale
prefix would mean the name `e054b-400m` was not fresh — it is, verified
empty).

**E-054a first-minutes verified + one loss recorded (11:45Z):** resume at
step 200,000 confirmed from the shipped log; the warm-restart val transient
arrived exactly as registered (0.03304 → 0.03471 @208k → 0.03409 @224k,
recovering, lr ≈9e-5). **The loss: continuing IN PLACE overwrites the 200k
finals** — by 224k the bucket's `temporal_e051.pt` already held the 224k
state, so the exact artefact that measured 0.0330 no longer exists anywhere
(§5.26 violated by this session's own design; impact small — the 400k final
supersedes it as the roll target — but the rule for every future
continue-in-place is: COPY the finals to a versioned name BEFORE the first
resume ship).

**E-054a VERDICT, 2026-08-27 23:50Z — 400k one-step ratio 0.02981: THE
BUDGET IS STILL PAYING, WITH DIMINISHING RETURNS — E-054b CONFIRMED.**
Final metrics (bucket, 169 records): step 400,000, val_zmse 0.63926 /
monitor persistence 21.44621 ⇒ **0.029808**, vs 0.0330@200k (−9.7% over
the doubled budget; trajectory 0.0330 → 0.0312@320k → 0.0300@388k →
0.0298@400k, decelerating). Against the two-sided registration: between
the poles, unambiguous in direction — more steps kept buying, at a rate
that says steps alone are no longer the cheap axis. The capacity rung is
the indicated move, and E-054b (launched 23:47Z on the freed us-west1-c
spot quota, node `e054b-400m`, first try) is its test. Schedule confound
carried from registration: "more steps" and "fresh LR" remain
unattributed. **Artefact lesson (§0.15, recorded the honest way): the
durable final is step 398,000, not 400,000** — tpu_train_s2's trainer
saves at CKPT_EVERY multiples DURING the loop and has NO post-loop final
save, so the exact-400k state existed only in device memory and died
with the node; the 55-min-idle node was manually deleted after the
shipper confirmed no further objects (self-reap had not fired — the
script sat in its ship loop after the trainer exited; a second launcher
gap to fix alongside the missing final save). The published roll target
is honestly named `head-weights-e051-398k-xl144zn-pentad-s0.pt`
(model-checkpoints-v1, 826.7 MB, opened and verified: step 398000,
K 144, 1024×16, seed 0, znoise 0.7); Δval between 398k and 400k is
~0.0001 in z-mse — immaterial for the roll, named anyway. Spot cost of
the whole continuation ≈ $23–27 incl. the idle tail.

**E-054b FIRST-MINUTES VERDICT, 2026-08-28 00:15Z — THE REGISTERED HBM
RISK FIRED: OOM AT THE FIRST TRAINING STEP.** Config echo confirmed the
build (399.948M params, 1280×20, K 144, batch 256, fresh, znoise
rel_pers 0.15116 — same relative dose class as E-051), then
`RESOURCE_EXHAUSTED: Attempting to reserve 5.09G ... 4.03G free` in
train_step. The 400M rung does NOT fit a v5e-4 chip at chunk/batch 256.
The exit-1 path worked as designed: final state shipped, node
SELF-DELETED (~28 min of spot, ≈$0.8). Options for the daytime decision,
recorded not chosen (the 01:45Z wake's own rule: no 2am improvisation):
(a) batch/chunk 128 — fits with margin but CONFOUNDS capacity with batch
(E-051 trained at 256); (b) gradient accumulation 2×128 = effective 256
— exact for a mean loss with no batch-coupled layers, needs a trainer
build + test; (c) a smaller rung that fits (~300M, 1152×18) — changes
the question from "2× params" to "1.5× params"; (d) remat/activation
checkpointing — same math, ~1.3× step cost, needs a trainer build.
(b) preserves the comparison exactly and is the principled fix; it and
(d) are subagent-buildable in a morning.

**E-054b RELAUNCH, 2026-08-28: OPTION (b) BUILT AND CERTIFIED, THEN LAUNCHED
TWICE — THE FIRST RELAUNCH WAS THE WRONG EXPERIMENT.** Gradient accumulation
(`--grad-accum N`, commit 6ec09b9) is exact for a mean loss with no
batch-coupled layers, and the certificate says so numerically:
averaged-micro-gradient vs whole-batch max rel **2.419e-07**, params after 3
steps agreeing to **8.693e-08**, `grad_norm` rel **0.0**, and the flag OFF is
bitwise the old path (1581 ops vs 1581) — `tests/test_grad_accum_jax.py`.
So `grad_accum 4 (micro 64)` is a memory decomposition of the SAME batch-256
step, not a batch change, and the E-051 comparison survives intact.

The first relaunch (07:08:38Z) then ran for an hour on a configuration that
was not this experiment. Its startup file had been rebuilt from the pristine
`ml/jaxport/tpu_train_s2.sh` with only D_MODEL/LAYERS/STEPS/TAG/GRAD_ACCUM
sed'd in, which reverted every other knob to the template's defaults —
`Z_ASSET` empty, K 24, lr 1e-3, stencil 1, ring '0', znoise 0, grad_clip 0,
halflife 40000 — so it was re-embedding an already-published Z (272,405,116
encoder forwards, ~3 h) to train a K=24 head with no noise and no clip. It was
caught by reading the `resolved knobs` line the launcher prints for exactly
this purpose, killed at 08:07:55Z, and relaunched at 08:10:33Z from a v3 file
sed'd out of the pristine template with the **full** E-054a knob block, whose
knob assignments were then `diff`'d against `/tmp/e054a_startup.sh` down to the
five intended differences (NODE, STEPS 400k→200k, D_MODEL 1024→1280, LAYERS
16→20, GRAD_ACCUM 4, TAG). Ledger row and the general lesson —
*sed the previous run's own file, never the template; `diff` the knob blocks
before launching* — in
[SPOT_LEDGER](https://blauewelt.github.io/earth/docs.html?f=ml/SPOT_LEDGER.md).

**E-054b FIRST-MINUTES VERDICT (relaunch), 08:29Z — CONFIGURATION CORRECT,
Z PULLED NOT REBUILT.** `resolved knobs: K 144 · steps 200000 · batch 256 ·
lr 4e-4 · 1280x20 · stencil 145 ring 'spiral:111-4444-0.71-0.5' · sched
expdecay halflife 100000 cooldown 0 warmup 2000 · znoise 0.7 · grad_clip 128 ·
grad_accum 4 (micro 64) · train_lon_hold none · seed 0 · tag 'e054b' · tensor
family4_na025_pentad_r2 (37e146384b) · codec run-415__pixelmae.pt · Z
'Z_8b639abe36_37e146384b.npy'`, and `Z VERIFIED: (3142, 86698, 32) float16,
16.24 GiB` assembled from 12 header-bounded chunks. Fresh run, step 0, no
resume. The OOM verdict itself is still owed — it lands at the first training
step, after the anomaly transform.

**THE ROLL IS DISPATCHED: #503 (E-051-roll — 365-day day-matched roll of
the 398k K=144 head, 3 starts/holdout year, dumproll, long/future rolls
for the replay battery, UNPOOLED transport keys default-ON — the first
roll ever to write E-055's new keys)** on the reserved fresh box
gpu-box-38116559, 23:52Z. K=144 roll cost unmeasured (#433's K=24 took
~13 h); >24 h token expiry ⇒ hand-harvest registered, §5.25 partials
ship as it goes. Controls: monthly _trainlon 0.939/0.939, pentad-K24
−0.499.

**E-054b FIRST TRAINING STEP, 08:37Z — NO OOM. THE REGISTERED HBM RISK IS
CLOSED BY GRADIENT ACCUMULATION.** `stage2_config` reads `params_M 399.948 ·
d_model 1280 · layers 20 · K 144 · batch 256 · grad_accum 4 · micro_batch 64 ·
stencil 145 · ring spiral:111-4444-0.71-0.5 · input_znoise 0.7 · grad_clip 128
· lr 4e-4 expdecay halflife 100000 warmup 2000 · train_lon_hold none · seed 0 ·
codec run-415__pixelmae.pt · z Z_8b639abe36_37e146384b.npy`, and
`stage2_monitor` reads `val_persistence 21.44621 · z_rms 5.17825 ·
input_znoise_rel_pers 0.15116`. Two things worth naming: the denominator is
**bit-identical to E-054a's**, so E-054b's ratio is directly comparable to
E-051's 0.0330 at the same 200k without any rescaling; and the relative noise
dose 0.15116 is the same class E-051 trained under, so the capacity contrast is
not confounded by the perturbation. Option (b) is vindicated — 400M runs on a
v5e-4 at the SAME batch-256 step E-051 used.

**RIDE-VS-CUT, RE-ARITHMETIC 13:30Z — THE RIDE CANNOT DELIVER THE BATTERY,
AND THE 06:50Z ANALYSIS THAT SAID IT COULD WAS WRONG ON TWO COUNTS.** Chris
answered "go ahead with 503" at 13:29Z, on the arithmetic below. Checking the
dispatch afterwards showed that arithmetic was wrong:

- **The skill phase is 441 steps, not 657.** 3,363 − 1,461 long − 1,461 future
  = 441. The `scored` mark therefore fired at **11:30:43Z**, not ~16:50Z.
- **`job_timeout` is 2400 minutes = 40 h**, from ~23:55Z 08-27, so the job is
  KILLED at **~15:52Z 08-29**. From 13:30Z that is 26.4 h; at the measured
  87.1 s/step (re-confirmed: 45,276 s / 520 steps) the run reaches step
  ~1,611 of 3,363 — **48% of the battery**. The long roll spans steps
  441→1,902 and its partial mark fires only on COMPLETION, so the job dies 291
  steps short of writing anything. `eta_all_s` in the run's own record is
  247,529 s = **68.8 h**, i.e. completion ~10:20Z Monday, 42 h past the
  timeout.

So riding buys **26 more hours at $0.320/h ≈ $8.4 and produces no artefact
beyond what was in hand at 11:30Z**. This was reported to Chris at 13:35Z as a
correction to the question he had already answered, with the recommendation to
cut and re-dispatch a battery sized to finish: the long and future rolls
shortened from 20 years to ~5 each (365 + 365 = 730 steps ≈ 17.7 h at the
measured pace) via `--long-months` / `--future-months`, which is a flag and
not a build. **The battery is no longer a formality** — the flat profile above
makes it the most informative measurement on the board, because if it confirms
replay it puts the MONTHLY 0.939 champion under the same question.

**#503 TIMING, MEASURED 2026-08-28 06:50Z — THE DECISIVE NUMBER ARRIVES
INSIDE THE TOKEN; THE BATTERY DOES NOT.** The roll is healthy and its own
progress record prices it: `skill` phase 240 of 3363 steps at
elapsed_s 20915 ⇒ **87.1 s per rolled step**, an ETA of ~75 h for the
phase as a whole. That total is misleading about WHEN numbers appear,
because the evaluator writes its partial at four marked points
(`rollout_spatial.py` 2482/3165/3395/3423-3438, then the single unmarked
final at 3451), and the one that matters is **line 3165,
`mark("scored")` — written when a head's SKILL scoring is complete.**
The step budget splits accordingly: **657 steps of skill** (9 starts =
3 per holdout year x 2009/2017/2023, x 73 horizon steps) and **2,922
steps of battery** (long roll 1,461 + future roll 1,461, i.e. 20 years
each at pentad). So:

- **the day-matched corridor AUC lands at step 657 ≈ 16:50Z 2026-08-28**,
  ~7 h before the job token expires at 23:52Z — it ships in the partial
  and is readable without any hand-harvest;
- **the replay battery needs ~71 further hours**, entirely outside the
  token: from 23:52Z the box computes blind (live pushes 401) and the
  artefacts would have to be hand-harvested from the runner on Sunday.

That is the decision to bring Chris rather than take: ride the battery
(~$23 of box time, lands Monday, blind for its last three days) versus
cut after `scored` and re-dispatch a battery-only run — the evaluator
already carries `--long-months` / `--future-months` (default -1 = full),
so a shortened battery is a flag, not a build, and an 80 GB box would
also cut the per-step cost. **Neither choice is free of a scientific
cost: §6 of the E-053 plan and the E-043b-PHASE rule both make the
battery MANDATORY before any rolled number is called forecast rather
than replay** — so whatever lands at 16:50Z is a corridor AUC awaiting
certification, and must be reported in exactly those words.

<a id="e-051-roll-read"></a>
### #503's DECISIVE NUMBER LANDED 11:30:43Z — 0.944, AND THE PROFILE SAYS REPLAY

The `scored` mark fired at **11:30:43Z**, five hours before the 16:50Z
estimate (the skill phase is **441** steps, not the 657 the earlier arithmetic
assumed: 3,363 total − 1,461 long − 1,461 future). Read from the partial, the
day-matched corridor AUC — the ONLY one comparable to an archived number — is:

| scope | `horizon_auc_daymatched` | `horizon_auc` (pentad axis) | n_px |
|---|---|---|---|
| corridor / corridor_trainlon | **0.944** | 0.888 | 30,158 |
| gate / gate_trainlon | 0.943 | 0.864 | 864 |
| window / window_trainlon | 0.950 | 0.887 | 86,698 |

`*_trainlon` equals its parent exactly, as the file's own `holdout_lon` note
predicts (`'0,0'` is the empty interval). Controls: monthly `_trainlon`
**0.939**, pentad K=24 **−0.499**. Transport bands, pooled / E-055 unpooled:
0.511/0.481 (5–90 d), 0.591/0.568 (95–180 d), 0.565/0.580 (185–365 d).

**Taken at face value this is the headline the pentad programme was built for**
— full-budget K=144 at pentad matching the monthly champion and erasing the
K=24 head's −0.499 collapse. **It should not be taken at face value, and the
run's own rows say why.**

| lead (days) | 5 | 10 | 15 | 30 | 60 | 90 | 150 | 245 | 305 | **365** |
|---|---|---|---|---|---|---|---|---|---|---|
| `msss_clim` | 0.971 | 0.960 | 0.736 | 0.949 | 0.948 | 0.942 | 0.938 | 0.946 | 0.945 | **0.946** |
| `acc` | 0.985 | 0.980 | 0.880 | 0.974 | 0.973 | 0.970 | 0.968 | 0.972 | 0.972 | 0.973 |

**The profile is FLAT.** Skill at a 365-day lead equals skill at a 30-day
lead. Forward physics decays with lead — that is what makes it forward
physics — and a flat accuracy-vs-lead curve is precisely the calendar-recall
signature the replay battery exists to detect (`docs/ML_BASICS.md`; the
E-043b-PHASE rule and E-053 plan §6 both make the battery MANDATORY before any
rolled number may be called forecast skill). Worse for the innocent reading:
`msss_pers` at h=73 is **0.966**, i.e. the model's MSE is 3.4% of
persistence's at a one-year lead, on a z-scored ANOMALY field with the
seasonal cycle already removed. There is no physical story for that.

**The mechanism is ordinary and was anticipated.** The skill phase rolls from
starts inside the holdout years (2009/2017/2023), but a 365-day roll from a
2009 start walks into 2010 — which is training data. Holding out a year holds
out that year's bins, not the future the roll travels into. That is exactly
the hole the battery's rolls-from-many-start-dates design probes.

**So the number is reported as: a corridor AUC of 0.944, UNCERTIFIED on two
independent counts** — no pentad validation-gate reference exists
(`gate {pass null, skipped true, certified false}`), and the replay battery has
not completed — **with a directly-visible flat profile that predicts it will
not survive certification.** And even the comparison to monthly is not a
"beat": +0.005 against a tier corridor sd of 0.0020 and a decision bar of
0.025 is a CONSISTENCY, not a level (§3b's ✅ forms).

**This roll carries no `per_channel`** — #503 was dispatched before E-058
landed. The SST column arrives with the next roll.

**#503 HEALTH AND A SECOND CAVEAT NOBODY HAD NAMED, read from its own partial
at 10:25Z.** The `rollout_spatial.json` on `ml-live-503` is well-formed and
progressing (`in_progress {heads 1, stage "started", at 00:50:18Z}`, `heads`
still `{}` — so the `scored` mark has not fired and the ~16:50Z estimate
stands), and E-055's new unpooled keys are live in it: `probe_unpooled` is
fitted on 1,240 train rows with all three holdout years excluded,
`fit_r_train_insample 0.755`, `fit_wall_s 3.9`, learned softmax attention over
the section's 266 pixels in place of `read_sv`'s mean — **the first roll ever
to write them**, and they carry the honest note that they have no reference of
their own and are not gated.

Two things in that file change how the 16:50Z number must be reported.

- **The roll is UNCERTIFIED, independently of the battery.** `gate` reads
  `pass null · skipped true · certified false`, because **no validation-gate
  reference exists at pentad cadence**: `e017_u1_s0`'s reference (auc 0.643,
  bands 0.47 / 0.375 / 0.492) was measured by `ml/rollout.py` over the MONTHLY
  family-3 axis in #217, and cannot certify a roll whose steps are a different
  length, whose starts are a different count and whose horizon bands span
  different durations. The file says it in its own words: *"Passing it here
  would be a certificate for an experiment nobody ran."* So whatever lands at
  16:50Z carries TWO caveats, not one — uncertified by the gate, AND awaiting
  the replay battery. They are separate, and both must be said. Owed:
  register a pentad reference in `GATE_REF_BY_CADENCE` once one is measured.
- **Read `horizon_auc_daymatched`, never `horizon_auc`.** The file's own
  cadence block is explicit: every `h` is an AXIS STEP, `horizon_auc` averages
  over h=1..73 and is therefore a function of THIS axis's lead sampling, while
  `horizon_auc_daymatched` averages the twelve leads (6,12,…,73 = 30…365 days)
  that ARE the monthly archive's — and is *"the ONLY one of the two that may
  be compared against an archived corridor AUC"*. The controls this number is
  read against (monthly _trainlon 0.939, pentad-K24 −0.499) are archived
  corridor AUCs, so quoting `horizon_auc` against them would be comparing two
  different lead samplings and calling the difference a result.

**THE E-051/E-054a ROLL (the decisive reading, next after E-054a lands):**
roll the **400k final** through `rollout_spatial` at pentad — the 12-month
day-matched corridor AUC on the trained-longitude corridor scope, against
monthly's +0.939 and pentad K=24's −0.499 — WITH the E-043b-PHASE context-end
battery (rolls from several different start years; a flat tracking-r-vs-lead
profile or calendar-pinned future peaks = memorised replay, and the number
is then reported as replay tracking, not skill). Head published to
`model-checkpoints-v1` from the bucket final via the sandbox; sroll dispatch
cloned from #433's shape (which proved the pentad roll path); `certified:
false` at pentad by construction. ~13–20 h on the 4090, ≈$4–7. Where
possible it reads transport through E-055's unpooled read-out (below) as
NEW keys beside the pooled legacy ones.

---

<a id="e-052"></a>
## E-052 · The field head: joint next-field prediction, deterministic then generative — BUILT AND CPU-VERIFIED 2026-08-25/26 overnight (no GPU arm dispatched)

**E-052 · build + CPU verification of the joint field head (axis-A deterministic
regression AND axis-B EDM diffusion) plus the probabilistic scoreboard · params:
toy configs 0.13–1.1M (real arms not yet sized) · stage: build / toy-eval only —
NOTHING trains on real data yet · data: synthetic laws `shift`/`bimodal`/`gauss`
(generative law documented in each generator's docstring) · arch:
OceanTokenizer(patch) + per-token TemporalCond (time only) + FieldDiT
(adaLN-zero, attention over space only) · steps×batch: toys 1.2k–16k × 8, CPU ·
resume: none.**

Chris approved the AR-vs-diffusion staged path (2026-08-25: *"Sounds good,
let's try this. Take things step by step ... please continue and thoroughly
test the diffusion implementation"*). Design argument:
[the deck](https://blauewelt.github.io/earth/ml/figures/ar_vs_diffusion.html);
experiment form and falsifiers:
[the E-052 plan](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E052_field_diffusion.md).
New files, nothing existing touched: `ml/probscore.py`, `ml/field_model.py`,
`ml/train_field.py`, `tests/test_probscore.py`, `tests/test_field_diffusion.py`.
Implementation by two Opus subagents to a main-session spec (§0b); every test
and number below re-run and read by the main session.

**(a) The test suites hold, mostly as EXACT identities (16/16 green, ~104 s
CPU).** probscore 7/7: ensemble CRPS against the Gaussian closed form (and a
longhand O(M²) reference pinning the sorted-member estimator), CRPS(M=1) == MAE
exactly, mse_sample == mse_mean + mean_var to 1e-10, spread–error calibration,
hand-computed Brier, NaN-holes == deleted-elements. field head 9/9, the
load-bearing ones bitwise: det mode at init IS persistence (`z_hat == z_t`,
`torch.equal` — the E-052 twin of "r_fore reads exactly 1.000000 at step 1");
diff mode at init has `D(x;σ) == c_skip(σ)·x` at six σs; tokenizer round-trip
identity with land holes; land values AND land output slots inert in both
losses; same-seed samples `torch.equal`, member m of an M=4 call == the M=1
call at the derived seed; 5-steps+resume+3 == 8 uninterrupted bitwise on all
parameters; result files atomic with `in_progress` (§5.25); NaN refused, never
written.

**(b) The bimodal head-to-head — axis B's microcosm, measured against analytic
targets** (`ml/runs/field/bimodal_headtohead.json`; law: x_{t+1} = x_t ±
PATTERN, one fair coin for the WHOLE field, so the conditional mean is
persistence and the two futures are field-coherent):

| read-out | det head (2k steps) | diffusion head (16k steps, M=16, NFE 29) | analytic perfect sampler |
|---|---|---|---|
| one-step ratio, per member | **0.9939** (the blur) | 2.0212 | 2.0 |
| one-step ratio, ensemble mean | — | 1.1464 | 1.0625 (= 1 + 1/M) |
| fair CRPS | 0.8854 (degenerate ensemble) | **0.4964** | 0.4427 (= half the det head's) |
| sign coherence (joint-law detector) | — | **0.9956** | 1.0 (factorized floor ≈ 0.031) |
| spread–error | — | 0.930 | 1.0 |

Reading: every slide-4 claim reproduces quantitatively. The deterministic head
is pinned at ~1.0 by construction; the sampler pays EXACTLY the conditional
variance per member (2.02 vs 2.0) yet wins CRPS by ~1.8× over the same head
scored as a degenerate ensemble; and the samples are JOINT — one field-wide
mode per member (coherence 0.996 against a factorized floor of 0.031), which
is the property no per-pixel head can have at any capacity. **(c) The shift
toy — axis A's microcosm** (`ml/runs/field/shift_det.json`; x_{t+1} = one-cell
eastward roll, a purely spatial law a neighbourless per-pixel head cannot beat
persistence on): field head ratio **0.0081** at 1.2k steps. **(d) The gauss
toy** (known conditional N(0.7·x_t, 0.5²)): sampled conditional mean slope
0.7343 vs 0.7, pooled sd 0.4670 vs 0.5 (test 8, M=64).

Toy numbers are CPU, n = 1, on synthetic laws — they verify MECHANISMS
(the estimators, the joint sampler, the exact identities), and are directions
for nothing beyond that. Real-data arms (E-052.1 then E-052.2, falsifiers
pre-registered in the plan) are NOT dispatched: `ml-train.yml` sits at the
25-input ceiling and the new trainer needs its own reviewed dispatch step —
the one piece deliberately left for a daytime decision. Cost of tonight:
$0 GPU; ~40 min CPU in-session.

**(e) UPDATE 2026-08-26 ~08:30Z — the JAX port, its parity certificates, and
the TPU launcher (Chris: *"go ahead with next steps (2) autonomously. Likely,
Jax compatibility will be required"* / *"Feel free to launch TPU jobs to
verify compatibility"*).** Three more files, all additive:
`ml/jaxport/field_model.py` (NNX mirror + two-way converter),
`ml/jaxport/train_field.py` (the TPU trainer: sharded batch via
`jax.device_put(NamedSharding)` per the 0f04c3e lesson, threaded host gather
with prefetch, token-chunked conditioner with REMAT — measured design
arithmetic in its docstring: without remat the conditioner's retained
activations are ~124 GB/chip at the train config and FLAT in chunk size,
because a reverse-mode `lax.map` stacks every chunk's residuals; with remat
~25 MB), and `ml/jaxport/tpu_train_field.sh` (verify/train modes; stages
tensor+Z from `gs://earth-tpu-staging`, and when absent ASSEMBLES them from
the releases exactly as `tpu_train_s2.sh` does and publishes the verified
bytes back to the bucket — §5.26). Parity gates
(`tests/test_jaxport_field.py`, F1–F9, CPU): det forward max|Δ| **7.15e-07**;
D(x;σ) **4.40e-07** over six σs; `edm_loss_given` rel **6.5e-08** with
injected (σ, noise); 8-step Heun `sample_from` **1.43e-06** with injected
x_init; gradient parity worst per-tensor rel **2.1e-07** over 64 tensors;
export round-trip all 68 tensors `torch.equal`; JAX resume bit-identical;
`--input-znoise 0` and remat-on/off both bit-identical on the update (the
remat gradient differs by ≤4 ulp, localized to exactly the 25 rematerialized
`cond.*` tensors, asserted as such). The parity surfaces
(`edm_loss_given`, `sample_from`) were added to `ml/field_model.py` with the
9/9 torch suite re-run bitwise-green. **E-052.1's config is fixed and
param-matched to the stencil tier: d_model 1024 × 10 blocks, heads 16,
d_cond 512 × 2, patch 4, K 144 (720-d span) = 200.4M** vs the 206.5M
pentad stencil head. Dispatch waits on the v5e quota (one v5litepod-4;
`e051-k144-full` holds it until ~14:00Z): first a VERIFY node (F-gates +
300-step real-data smoke on device, ~$5), then the E-052.1 train node — its
§0d dispatch entry will be written at dispatch.

**(f) 09:20–10:20Z — the spot shortcut was tried and the pool is still the
lemon it was on 08-23.** Preemptible v5e quota turns out to EXIST alongside
the on-demand 4 cores (useful permanently: a spot node can run beside an
on-demand one when the pool is healthy). But two `e052-verify` spot creates
in a row returned the SAME host (identical external IP) flagged
`UNHEALTHY_MAINTENANCE` at creation, and it never ran its startup script —
zero bucket writes in 32 minutes where the launcher's first publish lands at
~20–25. Deleted both; cost ~$1–2. Verdict: the us-west1-c spot pool is still
serving in-maintenance hosts (third and fourth sightings after E-051's two on
08-23) — do not re-try it today; the verify node waits for the on-demand slot
when `e051-k144-full` self-deletes.

**(g) UPDATE 2026-08-26 ~23:15Z — VERIFY PASSED (fourth node, 22:09Z boot),
and what the four rounds each caught.** Round 1 (16:16Z–20:00Z, three nodes):
all self-inflicted tooling — an incomplete rebake left `__BUCKET__` literal
in the startup script (nodes ran blind; placeholder guard now in tpu_box),
and a v5litepod-8 default against the 4-core grant read as a stuck quota.
Round 2 (21:18Z): the REAL defects begin — the pentad npz carries no ys/xs
(pixels object derived in-sandbox from X.npy's own ocean mask, 86,698 = Z's P,
RAPID at 26.5°N/266 px, now bucket-staged and defaulted); `measure_sigma_data`
built a ~409 GB ctx stack (host OOM-killed both legs; row-pair fix in both
trainers); the parity gates ran JAX-on-TPU against torch-on-CPU (device
numerics as port failure; gates now CPU-pinned). Round 3 (21:41Z): device OOM
— the f32 ctx is 6.4 GB/chip at the real geometry; ctx now ships at SOURCE
precision (f16, exact) with the upcast chunk-by-chunk inside `chunked_cond`.
Round 4: **PASS — gates 3/3, det leg rc 0 (300 steps, final ratio 0.9152,
loss falling), diff leg rc 0 (sampled eval runs end to end), params
200,406,016, sigma_data 4.4712, train/val 2779/219 windows, and the node
self-deleted on success — the full lifecycle is now proven.** Cost of the
day's verification: ≈$25–30 TPU across seven nodes.

**THE PACE RESETS THE BUDGET.** Steady state measured from the metrics tail:
15 steps in 46.9 s = **3.13 s/step** (batch 8 windows = 693,584 pixel-targets
per step). The planned "200k steps like the stencil tier" would be 7.2 days
and ≈$835 — and it was never the matched comparison: a field-head step sees
~1,350× more pixel-targets than a stencil step, and 200k steps is 576 epochs
of the 2,779 train windows. Step counts do not transfer across the two
architectures; matched WALL/COST on the same hardware does. E-052.1 therefore
trains **24,000 steps ≈ 21 h ≈ E-051's wall-clock** (≈63 epochs), halflife
rescaled to the steps/5 convention.

<a id="e-052-1"></a>
**(h) E-052.1 DISPATCHED 2026-08-26 ~23:20Z — `e052-1-train`, v5litepod-4.**

**E-052.1 · joint deterministic field head on the real pentad substrate — the
clean axis-A ablation (space moves into the head's attention; the predictand
becomes the whole next field) · params 200,406,016 (param-matched to the
206.5M K=144 stencil head) · stage: field head on the frozen run-415 pentad Z
(`Z_8b639abe36_37e146384b`, [3142, 86698, 32] f16) · data:
family4_na025_pentad_r2 (37e14638…) + the derived pixels index · arch:
OceanTokenizer patch 4 → 5,875 ocean tokens · TemporalCond K=144 (720-d span,
the E-045 lesson) × d_cond 512 × 2 · FieldDiT d_model 1024 × 10 × **8 heads**
(the first launch tried 16 to mirror the stencil tier and OOMed the chip —
attention scores are [B, H, 5875, 5875] and doubling H is +2.2 GB the 16 GB
HBM does not have; heads don't change the param count, and 8 is the exact
configuration the verify measured training at 3.13 s/step) · adaLN-zero,
residual prediction (zero-init ⇒ exact persistence at step 0) ·
det mode · steps×batch 24,000 × 8 windows, lr 3e-4 expdecay halflife 4,800
warmup 2,000, input-znoise 0.7 (context only), seed 0 · holdout: target-bin
years 2009/2017/2023 (train_stage2's rule) · resume: ckpt_latest.npz on
gs://earth-tpu-staging/runs/e052-1-train/, ship every 10 min, ckpt_every
1,000, stall watchdog 120 min (first ship ≈78 min at 3.2 s/step) · code
88cad0b · cost: ≈22 h ≈ **$106 on-demand** (spot if the pool stops serving
lemons: ≈$40).**

TL;DR — the stencil head asks "what happens at THIS pixel given its 145
concatenated neighbours"; the field head asks "what does the whole window do
next", attending over 5,875 ocean-patch tokens. CONTROLS at K=144/720-d span
on the same Z: #478's one-step ratio **0.0820** (20k stencil steps) and
E-051's full-budget **0.0330** — read against this run at matched WALL-CLOCK
(this run ≈ E-051's), never at matched step count (see (g)). **Falsifier
(pre-registered in
[the E-052 plan](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E052_field_diffusion.md)):
if joint spatial attention ≈ per-pixel stencil-concat at matched conditions,
axis A buys nothing here — the 145-slot concat was never the bottleneck — and
the diffusion case must rest on axis B (E-052.2, same backbone in diff mode,
follows this run's read-out).** Chris's order (2026-08-26): *"Let's implement
the stage-2 head first, then the full move."* Overfitting watch: 63 epochs of
2,779 windows under znoise 0.7 — the val curve at eval_every 2,000 is the
read; a rising val ratio with falling train loss ends the run early via the
resume machinery, not by letting it ride.

<a id="e-052-1b"></a>
**(i) E-052.1b DISPATCHED 2026-08-27 ~12:30Z — the low-LR parallel arm
(`e052-1b-lr1e4`, v5litepod-4, us-central1-a).** Chris, reading the live
curve at step ~7k: *"The curve looks as if a smaller learning rate (from the
start) would help. Would you mind trying a parallel run?"* The curve supports
it: train loss is jagged rather than descending smoothly, grad_norm has RISEN
under a decaying schedule (2.0 at step 240 → 5.0 at 6,720; the toy runs sat
near 2 throughout), and the val ratio near-flatlined 0.580 → 0.578 across
evals 2 and 3 — all consistent with a too-hot start for this optimisation
surface (which is NOT the stencil head's: 3e-4 was inherited from that tier).
**One knob moves: lr 3e-4 → 1e-4.** Same warmup 2,000, same halflife 4,800
(the schedule keeps its shape), same seed 0, same 24,000 × batch 8, same
geometry, same code (dc37494). Same falsifier and controls as (h) — whichever
arm's VAL curve is better at matched step becomes E-052.1's reported number,
and the loser is the LR ablation. Zone note: both us-west1-c slots are
occupied (this experiment on-demand; the E-051 continuation on spot), so this
arm went zone-shopping — yesterday's cross-zone 429s were the v5litepod-8
request bug, and GCP's own error text ("Limit: 4 in zone us-central1-a") was
the evidence per-zone grants exist. Measured 12:25Z: us-central1-a has NO
capacity (code 8, spot AND on-demand — a capacity error, not quota);
**us-west4-a spot served a healthy node first try** (boot beacon at +4 min).
Runs on `us-west4-a` spot: ≈$40, ≈22 h, preemption covered by the
ckpt-every-1,000 resume machinery.

**(j) HARVESTED 2026-08-28 ~03:45Z — THE AXIS-A FALSIFIER FIRED, both arms.**

**Arm A (lr 3e-4), complete at 24,000 steps.** Val one-step ratio trajectory
(eval every 2,000): 0.684 → 0.580 → 0.578 → 0.560 → **0.5525 best at step
10,000** → then monotone worsening to **0.5808 final at 24,000** while train
loss fell 13.3 → 7.18 — the textbook overfitting scissors, on schedule for
2,779 train windows under a 200.4M-param head; input-znoise 0.7 slowed it,
did not stop it. Cost ≈$108 on-demand. Finals on
`gs://earth-tpu-staging/runs/e052-1-train/` incl. the torch export
`field_e052_1.pt` (produced at harvest in the sandbox — the on-node export
is best-effort-skipped in the JAX-only venv; note the bucket retains only
`ckpt_latest`, so the best-val step-10k weights are NOT retained — a
periodic named-checkpoint option is the follow-up if best-val weights ever
matter).

**Arm B (lr 1e-4), STOPPED EARLY at step ~15,600 of 24,000** (≈$36 spot,
≈$18 saved): its question — does a cooler start fix the optimisation? — was
answered by eval 7: best **0.6413 at step 4,000**, then monotone worsening
(0.676, 0.656, 0.680, 0.687, **0.6995 at 14,000**). The lower LR overfits
EARLIER and from a WORSE bottom; the jagged arm-A loss was a symptom of the
data/capacity ratio, not of step size. Resumable from its shipped ckpt if
ever wanted.

**VERDICT against the pre-registered falsifier
([plan](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E052_field_diffusion.md)):
joint spatial attention does NOT match per-pixel stencil-concat at matched
conditions — it is ~7× worse (best 0.5525 vs #478's 0.0820 at comparable
budget, vs E-051's 0.0330 at matched wall-clock), and the binding constraint
is DATA, not optimisation: a field head sees 2,779 training examples where
the stencil tier's per-pixel framing sees 2,779 × 86,698. The 145-slot
concat was never the bottleneck.** Consequence, per the plan's own branch:
the case for the generative move (E-052.2) can no longer ride on geometry —
running EDM diffusion on THIS backbone would measure distributional quality
around a mean that is 7× off, so E-052.2 in its planned form is NOT
dispatched pending Chris. The constructive options, in this session's
preference order: (1) **E-052.p** — σ-conditioned STENCIL head, axis B on
the geometry that works, cheap and the cleanest remaining test of "does a
distribution head earn its NFE"; (2) shrink/regularise the field head
toward the data (fewer params, patch 8, stronger augmentation) — only worth
it if the joint-field property is wanted for its own sake (multi-step
coherence), which the one-step scoreboard cannot show; (3) E-052.2 as
planned, accepting the mean gap, to read CRPS/spread mechanics on real data
anyway. Total E-052 real-data spend: ≈$144 across both arms + $17 verify.
## E-050 · Warm-start quantization: the trained continuous codec, lattice switched on — #485 DISPATCHED 06:32Z 2026-08-26 (approved by Chris 2026-08-25 ~15:30Z, b3ee36a)

**07:25Z — FIRST-MINUTES VERIFICATION PASSED ON ALL FOUR CHECKS** (#485,
gpu-box-32966687, ml-live-485): *(1)* the warm start is real — config carries
`fsq_warmstart_from: "run-480.pt@200000"` and the resumed record reads
`at_step 200000`; *(2)* the rebased fits at +50/+200/+2000 are the ANTI-#482:
`prequant_rms` exactly 1.0 and `prequant_std_med` **0.971 / 0.995 / 0.963** —
richly input-dependent where the cold start slid 0.638 → 0.005 by 7.5k — with
fitted scales O(1.5–1.7) and the spec nearly unchanged between refits (the
settled-distribution premise, observed); *(3)* loss_rec jumped to 0.310 at
lattice-on (the quantization tax, expected) and is recovering — 0.281 by step
206,700 against the parent's 0.229; *(4)* the step-202,500 probe reads
NUMBERS through the lattice (z_vs_persistence 37.8%, linear_r_deseas 0.44,
temporal_r_deseas 0.43) — no NaN, no collapse signature. Pace ~0.2 s/step →
60k warm steps land ≈10:30Z + the probe ladder. Falsifier A has not fired at
the point where #481/#482 were already dying; the registered verdict remains
the decoder-ceiling audit (Falsifier B).

**E-050 · resume the finished E-049a continuous d_z-6 pentad codec with the FSQ
lattice + ln bound switched ON at resume (`--fsq-warmstart`, 6c83ed3) · params
37.956M (measured identical both arms — lattice and bound are parameter-free) ·
stage encoder (resume + 60k warm steps) · data family4_na025_pentad_r2 · arch
512×12 d_z 6 patch 1, FSQ [8,8,8,5,5,5] = one 2¹⁶ token per pixel-bin, ladder
auto REBASED to +50/+200/+2000/+20000/+60000, bound ln · steps×batch
260,000×512 (200k resumed + 60k warm) · resume run-480 (required; promoted to
`run-480__pixelmae.pt` on model-checkpoints-v1 this morning from Actions
artifact pixelmae-480 — the workflow's own release publish never ran, only the
artifact upload; §0.2 again) · recipe `f4r2-40M-dz6-fsq65k-warm`.**

TL;DR — every cold-start codec-side FSQ run collapsed or degenerated (e048a,
#481/#482 constant encoder; run-455 sign code) while quantization on an
EXISTING z works (A9 0.4916) and lattice z out-forecasts continuous z at the
registered instrument (E-046/#477: 0.4394 vs 0.5036). Does giving the lattice
a TRAINED encoder — directions already spread — produce the road-B token
without the collapse? HYPOTHESIS + FALSIFIERS registered in
[the E-050 plan](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E050_warmstart_fsq.md):
(A) the #482 signature reappearing in the warm fits (std_med → ~0) kills the
warm-start repair and points at the bound (RMS-only, gated on the corpse
diagnosis — #481's ~10k corpse is already published as
`rescued-orphan-latest-482.pt`, #482's ~15k corpse publishes on
gpu-box-42005419's next job); (B) any fast channel at ~100% FVU at the
decoder ceiling on Argo-free bins = 16 bits cannot carry a bin even warm, and
the pre-registered next rung is TWO tokens (d_z 12), never a bigger vocab.
Declared confound: 260k total vs #480's 200k — the falsifier is absolute (the
9–19% FVU band); a PASS near the band edge buys a 260k continuous control.
First-minutes health: the FSQ WARM START banner (`run-480.pt@200000`), fit
prequant_rms 1.0 with INPUT-DEPENDENT std_med, effective bits ≫ the ~6-bit
sign-code floor. Runs pinned to gpu-box-32966687 (the parent's own box,
running idle at dispatch — tensor warm). COST ~4.5 h train + probes ≈ $2–3.
n = 1 at a new configuration (§3b): the first result buys its pair before any
number is a level.

<a id="e-049"></a>
## E-049 · Road B: one 16-bit token per pixel-bin, judged by a non-linear decoder — DISPATCHING 2026-08-25 (Chris: *"continue with road B and do that very diligently, and test it very well with a decoder (a non-linear decoder)"*)

**E-049a** · continuous d_z-6 width-tax control, fresh per-bin pentad codec · params
37.956M (measured) · stage encoder · data family4_na025_pentad_r2 · arch 512×12 d_z 6
patch 1 · steps 200k×512 · resume none · recipe `f4r2-40M-dz6`.
**E-049b** · ROAD B — the paper-faithful token: d_z 6, FSQ [8,8,8,5,5,5] (2^16 = 64,000
implicit codebook, ONE 16-bit token per pixel per 5 days), fitted ladder
(`auto`, refits at steps 50/200/2000/20000) behind a NEW intrinsic bound
(`--fsq-bound ln`, LayerNorm no-affine on the pre-quantization activation) · params
37.956M (identical to a — the lattice and bound add zero parameters) · stage encoder ·
data family4_na025_pentad_r2 · arch 512×12 d_z 6 patch 1 · steps 200k×512 · resume
none · recipe `f4r2-40M-dz6-fsq65k`.

**Hypothesis** (ml/plans/E049_roadB_token.md): a typical Argo-free pixel-bin's live
state (~8 fast channels, ~128 stored bits) fits in ONE 16-bit token well enough that
the DECODER CEILING — the E-019b1-style decoder-only retrain against the frozen
tokens, the strongest non-linear decoder we run — reconstructs the fast channels on
Argo-free bins at FVU within the month-block codec's accepted 9–19% band (E-047
Tier-1, the programme's reference price for a real big squeeze). **Falsifier,
registered now:** any fast channel at ~100% FVU at that ceiling = 16 bits cannot
carry a bin, the configuration is dead, and the pre-registered next rung is TWO
tokens per bin (d_z 12, 32 bits) — not a larger vocabulary. The Argo-carrying 8% of
bins are EXPECTED to reconstruct badly (they already collapse at d_z 32: cur_speed
112%, ssh 86%) and are reported, never scored against the hypothesis. **Attribution
by construction:** (a − run-415) = the width tax, (b − a) = the quantization tax;
without cell a, neither a good nor a bad b could name its disease.

**Why the bound is part of the dispatch and not a refinement:** run-455 measured what
an unbounded per-bin FSQ codec does (pre-quantization |v| ~ 3e4 vs R = 2 — the
8-level lattice worn as a ONE-BIT sign code), and e048a2 measured that the fitted
ladder alone closes the collapse but not the drift (std_med 0.73→20 across 28k
steps). At 16 nominal bits a sign-code degeneration leaves 6, and the run would
re-measure a known disease. This takes the first branch of the 08-25 design fork
(intrinsic bound), for the road-B line ONLY — E-048's w6s3 arm stays held on its own
fork. First-minutes health signal: **`prequant_rms` = 1 and FLAT at every fit** (not
std_med — LayerNorm fixes the vector's rms; measured 0.9998/0.38 on the CPU toy).
Effective bits are MEASURED, never inferred: `ml/fsq_usage.py` (new) recovers the
digits from any z cache and reports per-dim entropy — the instrument the run-455
lesson demands (nominal 96 bits, worn ~32).

**Code at this dispatch** (all CPU-validated; tests/test_e049_fsq_bound.py 7/7,
config guards 5/5, workflow guards 5/5 with the two new recipe-only keys, E-048
ladder tests green on the non-JAX checks): `fsq_bound` in ml/model.py + ml/train.py +
codec_from_ckpt's known set (old loaders REFUSE a bounded checkpoint rather than
dropping the bound; the JAX trainer refuses the flag, convert.py refuses by value);
workflow wiring for `fsq_auto_step`/`fsq_bound`; ml/fsq_usage.py. **Known instrument
gaps, stated before dispatch:** the collapse guard reads NaN on lattice z (the E-046
gap — arm b is monitored by prequant_rms + probe instead), and the recon audit
scripts need the pre-audit adaptation listed in plan §4 before Tier-1 runs (the
float16-accumulator overflow in recon_eval.py is silent and fatal at family-4 shape;
the Argo-bin split scorer does not exist yet). That adaptation is the named work item
for the ~20 h training window. **This wave trains NO stage-2 head** — token-input and
token-output heads are separate dispatches gated on the audit. §3b applies in its
harder form: new codec, new width, new bound = a tier with no replicate band; first
results are directions.

**Scale numbers** (rule 6): 37,956,471 params · batch 512 · 200,000 steps · the
pentad tensor's train pool (~everything outside holdout years at patch 1). Cost:
~20 h / ~$6 per arm, two arms.

**DISPATCHED 2026-08-25, code 72a0921 (both arms):** **#479 (E-049b road-B token
codec)** picked up immediately by `gpu-box-40623952`; **#480 (E-049a d_z-6
continuous control)** queued on `gpu-box-32966687` behind #477 (E-046-HEAD seed 1)
— that box holds the pentad tensor warm from the E-046 work. Plans published
(`plan-479/480/481.json`, 200k at the trainer's 3e-4 default).

**#479 DIED IN 2.5 MINUTES ON A DEAD GPU, AND THE GUARD PAID FOR ITSELF** (caught
by the scheduled 50-min first-minutes check): `nvidia-smi` on gpu-box-40623952
read "Unable to determine the device handle for GPU0 … No devices were found" and
the wrong-hardware guard refused to train on CPU — exactly its job, cost ~$0.01.
The trap worth recording: **the runner reported online + idle to GitHub while its
GPU was already dead** — runner liveness says nothing about the device, and the
install-step guard is the only honest hardware check we have; a dispatch IS the
health probe. Vast 48478310 now shows offline/stopped (a lemon; left stopped —
nothing sole-copy on it, #472 runs elsewhere on gpu-box-48383989). Recovery:
started 48478309 (`gpu-box-42005419`, the E-047a box, pentad tensor warm — the
build step was a cache hit) and re-dispatched as **#481 (E-049b re-dispatch, road-B
token codec)**: GPU guard PASSED, training phase opened 11:36Z, and the live
config record verifies the recipe end to end — d_z 6, fsq_levels 8,8,8,5,5,5,
fsq_ladder auto, **fsq_bound ln**, params_M 37.956, family4_na025_pentad_r2.
Step-50 fit line (`prequant_rms` = 1) checked at the next wake-up.

**#481 · 12:48–13:10Z — THE BOUND WORKS, AND THE COLLAPSE GUARD FALSE-FIRED ON THE
FIRST INSTRUMENT THAT COULD SEE LATTICE Z.** Two findings, one per direction:

*(a) The intrinsic bound's first production reading is exactly its spec.* All three
early fits ran (50/200/2000; 6/6 dims fitted, all exponential):
**`prequant_rms` 1.0 at every fit** — the invariant, to the digit — and fitted
saturation radii **O(1)** (0.39–3.3), where run-455 sat at |v|~3e4 and e048a2's
unbounded std_med ran 0.73→20. The drift disease is closed by construction, on the
real tensor, at batch 512. Loss_rec ~0.33 at 9k steps; the encoder was
INPUT-DEPENDENT at step 2000 by the fit's own sample (per-dim std 0.355 across
4,096 vectors; e048a's true collapse read one distinct vector per 1,024).

*(b) The guard then killed the run at step 10,000/200,000* — strikes
`r_des −0.024 @ 7500, −0.000 @ 10000` against the 0.05 threshold. This is the
**false-positive twin of #455's all-NaN blindness**: no FSQ codec has ever been
under this guard (the pooled probe read NaN on unbounded lattice z all of #455),
so the bound is what made the instrument read numbers at all — and its 0.05
threshold, calibrated on continuous pooled z, has an UNMEASURED healthy range on
16-bit lattice z. §3's own rule ("arming a guard on a new instrument requires
measuring its healthy range first") applied in reverse: the guard is disabled for
this arm (`collapse_r "0"`, new recipe-only key, wired + guarded by
test_workflow_config 5/5), with the REPLACEMENT monitor named in the recipe: the
fit schedule extended to 50/200/2k/20k/60k/120k/180k so `prequant_std_med`/`rms`
publish across the whole run (late fits are near-no-ops — the incumbent lattice
wins ties and only strict MSE gain moves a dimension), and a constant encoder
reads std_med ~0 there. Note r ~ 0 at 5% of training says nothing the falsifier
cares about: run-455's pooled instruments read garbage all run while its verdict
head was trending BELOW controls at cut-off. Cost of the lesson: ~1.6 h ≈ $0.5.
**Re-dispatch = #482 (E-049b, 3rd copy)**, same box (healthy, tensor warm), with
`job_timeout` 1800 — #481 also measured the pace, ~0.49 s/step ⇒ 200k ≈ 27.2 h,
OVER the 1500-min timeout it was dispatched with (the #456 class, caught before
it spent 25 h this time). #480 (the continuous d_z-6 control) keeps its guard
armed: continuous pooled z is the instrument the 0.05 was calibrated on, and if
d_z 6 continuous ALSO probes under it early, that is a measurement about d_z 6
worth having rather than a config to pre-empt.

---

<a id="e-047"></a>
## E-047 · The month-block codec — fusion vs selection, DISPATCHED 2026-08-23 ~12:00Z (Chris's direction: combine multiple 5-day points into one embedding, with the time labelled properly)

One embedding per CALENDAR MONTH built from a 7×40 cell grid over the pentad tensor
(k_max 7, learned within-block time-offset embeddings on both encoder cells and decoder
queries — decision (b), one symbol one meaning; padded/missing cells ride the
unobserved-cell path, so the mid-month Argo anchor fuses with its pentads into ONE
uniform representation — the architectural form of the r3 Argo-fill repair), ctx carrying
the CONTINUOUS block-centre phase (the labelling fix, at the codec level from birth).
**Hypothesis: fusion beats selection** — the 20k stage-2 head on this z beats E-045-A2a's
0.0721, which answered the cadence question by SELECTING one bin per month; falsifiers
and the two deliberately-skipped per-bin evaluations are in the dispatch doc and
`ml/plans/E047_block_codec.md`. The block axis reads as MONTHLY downstream, so the
follow-on roll is day-matched comparable to the monthly archive while built entirely
from 5-day inputs. E-047a = #448 (40M, d_z 64) on fresh box gpu-box-42005419; **E-047b = #449 — the
CHINCHILLA-ANCHORED sibling** (measured 126.943M = 1.006× the tensor's own anchor of
2,524,251,443 values/20, 1024×10, head_dim exactly 128, d_z 128, batch 256-estimated) on
fresh box gpu-box-40623952, dispatched IN PARALLEL per Chris's sizing question and
parallelization mandate (~$4 + ~$10). The pair answers fusion-vs-selection AND
40M-vs-anchor AND d_z-64-vs-128 in one overnight wave; 20k heads + block-decode rolls
(Tier-2 roll reconstruction landed at aacb014 — every cell of each predicted block scores
against its own bin, cell-level persistence baselines, dumps stay z-states) follow. Code:
timeblocks module + wiring at 2e03913/41086a3, smoke-tested end to end on CPU;
`time_block` is a recipe-only key (`f4r2-40M-monthblock`, d_z 64).

**12:50Z 08-26 — THE FUSION-VS-SELECTION VERDICT LANDS: #483 (E-047-HEAD, 6th
dispatch) GREEN 09:52Z — THE MONTH-BLOCK HEAD READS 0.2127 WHERE SELECTION READ
0.0721. FUSION DOES NOT BEAT SELECTION AT THE REGISTERED INSTRUMENT.**

*(a) The registered verdict.* 20k one-step z ratio on the month-block z
(e047a-tpu-60k codec, d_z 64, K=24 blocks = the same 720-d span as A2a):
`z_mse_model 0.58485 / z_mse_persistence 2.74977` = **0.2127**
(`probes-483.json`, archived; sha `fc5f62d`). The dispatch registered two
branches — "clearly below E-045-A2a's 0.0721 = fusing a month of pentads beats
selecting one bin per month; parity = selection suffices" — and the measurement
took the unregistered third: **2.9× ABOVE the selection arm.** By the
pre-registered reading, fusion does not beat selection; at this codec and
budget the fused month is the substantially harder one-step substrate.

*(b) The caveat that keeps (a) honest, recorded not used.* The ratio scores
each head against ITS OWN z-space's persistence, and month-block persistence
is a far stronger baseline (2.75 block-units vs ~21.3–21.4 in the per-bin
spaces) precisely because fusing a month of pentads averages away the fast
variance a per-bin head gets credit for predicting. The instrument was chosen
at dispatch knowing the spaces differ; the caveat qualifies the size of the
gap, not the direction of the verdict.

*(c) Secondary read-outs agree in direction.* The head-features transport
k-fold reads **0.394 [0.306, 0.475]** (n=1459, RMSE 3.7 Sv) — far under every
per-bin stage-2 head at this geometry (#427 0.583, #478 0.679) and under the
same bundle's codec-independent controls: raw-3×3 head 0.693 [0.633, 0.746],
wind head 0.690 [0.62, 0.751]. `chan_t+1` skipped by design (which cell stands
for the block is undecided); pooled probes excluded by protocol; bundle is the
known 4-file month-block set.

*(d) Scale block* (rule 6, source-written): 211,352,640 params · batch 256 ·
20,000 steps · 39,534,288 windows · K 24 · stencil 145. Training itself was
textbook: grad-clip 128 with clip_frac 0 throughout, val_zmse 0.597 at 20k and
flattening, LR 7.3e-4 at cut-off.

*(e) Ops.* Job 23:33→09:52Z ≈ 10.3 h H100 ≈ $21, ~6.7 h of it the block
re-embed (516 blocks at ~47 s) — the 00:50Z finding priced in. **The §5.26
publish watch resolves NEGATIVE: no `header_t-516` chunks landed on
embed-cache-v1** (all steps green; the partial-publish path simply never
covered the block key), so the month-block Z's only copy is gpu-box-48254133's
disk (Vast 48632885, stopped after drain, queue checked). At a negative
verdict that is priced as a ~$15 re-embed if the fusion axis is revisited, not
a rescue item.

*(f) What it means for the programme.* With the E-045 factorial closed (span
is the axis, 0.0820 at 5-d steps), E-047's fusion was the named candidate for
"the missing ingredient" should E-051's full-budget 2-year-context roll still
read E-044b-class negative. This verdict removes month-block fusion — at this
codec, d_z and budget — from that list; E-051's roll reading now stands alone.
The block-decode roll (Tier-2 machinery at aacb014) is NOT dispatched on a
negative one-step verdict. Whether to spend on a fusion variant (block codec
at d_z 32; the 126.9M block codec's z, pixelmae-472; fusion at the head rather
than the codec) returns to Chris. n = 1 at a tier with no pair (§3b): the
direction is measured, no number here is a level.

**RE-DISPATCH, 16:29Z: E-047a = #450, E-047b = #451.** #448/#449 both died of CUDA OOM
mid-training on their 24 GB cards — a month block is 282 encoder tokens against a per-bin
sample's 42 and attention is quadratic in that — and #448's probe path additionally
refused block-z (the collapse guard would have starved; block map plumbed through
probe_now, plus a probe row-keying fix, 6bd0ca4). Batches cut 512→128 (a) / 256→64 (b) as
a 24 GB measurement, recipes updated in place with the arithmetic. Both healthy at 23k/9k
by 19:30Z: 0.412 s/step (a, →200k ≈ Mon 15:20Z) and 0.683 s/step (b, →200k ≈ Tue 08:30Z).

**14:25–14:40Z 08-25 — #482 (E-049b, 3rd copy) CANCELLED AT STEP ~15k: THE ARM-B
CONFIGURATION COLLAPSES TO A CONSTANT ENCODER, AND #481'S GUARD KILL IS RE-READ AS A
TRUE POSITIVE.** The correction first, because the log said otherwise this morning:
**the "false positive" reading of #481 is WITHDRAWN.** What stands from that block is
only that the guard's instrument was uncalibrated on lattice z; the kill itself was
right. The evidence, three instruments agreeing on #482 (collapse_r 0, so it ran on
past where #481 died and showed the whole trajectory): the step-0 probe is healthy
(r 0.15, z persistence MSE 0.141); by step 7500 persistence MSE is **0.000915**
(~150× down) and r reads exactly −0.000; the step-2000 fit's `prequant_std_med`
drops to **0.005** (0.638 → 0.371 → 0.005 across the three fits — the replacement
monitor doing precisely what it was armed for, with the fit's R_min 0.10 showing
the lattice chasing a collapsing distribution); and the step-15000 full probe dies
of **ZeroDivisionError — persistence MSE exactly 0, i.e. z constant over time**.
The bound itself held throughout (`prequant_rms` 1.0 at every fit): this is a
DIRECTION collapse under an intact scale, e048a's disease in a new coat — and
run-455 (d_z 32, unbounded) did NOT collapse to constant, so the new variables are
d_z 6, the LN bound's mean-centering (2 of 6 DoF), the 5-level dims, and the early
refits. Also measured, a correction to this morning's timeout arithmetic: **0.252
s/step** (wall_s 3781.4 at step 15,000) — the 0.49 estimate had conflated the
standardization pre-pass; 200k is ~14 h and fit even the original 1500.

**Next, per §4.13 (fix the class, no 4th blind copy):** *(i)* #482's 15k checkpoint
survives on gpu-box-42005419's disk (box STOPPED, disk kept; the rescue step
publishes it on the box's next job) — the e048a diagnosis playbook applies: distinct-z
count, where between to_z and the lattice the constancy is born (pre-LN constant =
encoder's doing; post-LN constant only = the centering's). *(ii)* The candidate
repair with the best mechanism story: **WARM-START quantization** — train the
continuous d_z-6 codec first (exactly arm A / #480, already queued), then enable the
lattice on resume from its checkpoint; needs a small guarded train.py affordance
(fsq keys are arch-defining, so --resume currently refuses the change — correctly;
an explicit opt-in flag, its own commit). This is A9's own shape moved codec-side:
head-side quantization worked ON AN EXISTING Z, and every cold-start codec-side FSQ
run here has now either collapsed (e048a, #481/#482) or degenerated (run-455's sign
code). *(iii)* The cheap alternative if diagnosis blames the centering: an RMS-only
bound (no mean subtraction). Cost of #482: ~1.1 h ≈ $0.3.

**20:35Z 08-25 — E-051 DISPATCHED (Chris: "yes, let's go straight to the big tpu
run"): THE FIRST FULL-BUDGET 2-YEAR-CONTEXT PENTAD HEAD, ON THE TPU TIER.**

<a id="e-051"></a>
**E-051 · the span-fixed K=144 pentad head at full budget · params 206.5M head +
fresh k_max-144 table · stage stage-2 · data family4_na025_pentad_r2 (sha-pinned
37e14638…) · arch 1024×16 over the frozen 38.0M run-415 pentad codec (d_z 32) ·
steps 200,000×256 · resume run-415 (stage 1 trains nothing) · framework
JAX/v5litepod-4 (tier 3b, G5-certified — NEVER pooled with torch numbers, §3b) ·
window stencil:145 ring spiral:111-4444-0.71-0.5 znoise 0.7 grad-clip 128 seed 0
sched expdecay(halflife 40k, warmup 2k, cooldown 0) · Z PULLED from the published
clean `Z_8b639abe36_37e146384b` (header-bounded assembly — the launcher's own fix
for the chimera class) · node `e051-k144-full`, bucket-keyed resume, on-demand.**

HYPOTHESIS: the E-045 factorial's context story holds at full budget and at
horizon — a 2-year-span pentad head trained to 200k is the programme's best
stage-2 substrate, beating the monthly champion line on the roll. REGISTERED
READINGS, in order of authority: (1) the 12-month ROLL (corridor AUC, on GPU
via sroll over the exported .pt head — a follow-up dispatch, not part of this
run) against the monthly pair's +0.939 trainlon and E-044b's pentad −0.499;
(2) the one-step ratio vs #478's 20k number (in-flight 0.0860 at 17.2k as this
dispatches; #427's 20k→200k flatline at K=24 predicts little one-step gain, so
a flat ratio here is EXPECTED, not a failure). FALSIFIER: if the 200k head's
roll still reads E-044b-class negative, 2-year context did not repair the
pentad roll and the missing ingredient is elsewhere (the fusion axis, E-047).
CAVEATS ON THE RECORD AT DISPATCH: #478 (the torch 20k twin at this exact
geometry) had not finished when this launched — its final lands ≈22:00Z and is
the config's certificate; the JAX tier is n=1 with no replicate band (§3b — a
first result here is a direction); and the roll-first staging was offered and
DECLINED by Chris in favour of going straight to the full run. COST: ~8–15 h
at v5e-4 on-demand ≈ $40–72 (spot declined — the 08-23 us-west1-c spot pool
served in-maintenance hosts twice); hard cap 40 h; stall watchdog 90 min;
checkpoints + torch-loadable .pt ship to the bucket every 10 min, so a
preemption or cap costs a relaunch under the same node name, never the run.

**21:35Z 08-25 — E-051 LAUNCH ADDENDUM: TWO NODES DIED IN SETUP (BOTH
SELF-REAPED CORRECTLY), THE THIRD IS TRAINING.** TL;DR — the big TPU run is
live on its third node; the first two deaths were a disk guard doing its job
and a real JAX trainer bug at K=144, now fixed on main. *(i)* Node 1
(~20:19Z): the launcher's disk guard refused 90 GB free vs the 120 the pentad
allocation needs, shipped its log, self-deleted (~$0.20). Fix: `WORK` on
tmpfs, remounted to 170G (189 GB RAM host). *(ii)* Node 2 (20:30–20:58Z,
~$4.50): HBM OOM at the fixed monitoring batch — `put()` staged the FULL
4096×144×145×32 fp32 array (10.95 GB) onto device 0 via `jnp.asarray` and
then re-sharded it, so device 0 needed the whole array PLUS its own 2.55 GiB
shard beside the replicated head + Adam state on a 16 GB v5e chip; XLA's
"563.88M free" is 16 GB minus exactly those terms. Invisible at gate-tier K
(the staged array is small) — a genuinely new K=144 failure. Fix on main
([0f04c3e](https://github.com/blauewelt/earth/commit/0f04c3ee0346f132720ec3a0a99fa178f1344197)):
`jax.device_put(numpy, NamedSharding)` slices on the HOST, no full-size
device allocation ever exists; dtype canonicalization measured identical; all
5 G5 gates re-run green. *(iii)* Node 3 (boot 21:07Z, training ~21:30Z):
staging clean — Z assembled from 12 chunks header-bounded and VERIFIED
(3142, 86698, 32) float16 16.24 GiB with monitor z_rms 5.17825 (the
continuous run-415 identity, checked by VALUE per the ledger rule), fresh-run
resume line on the record, val_persistence 21.446, znoise 0.7 = 0.151×
sqrt(val_persistence) (vs the monthly anchor's 0.398× — pentad regime,
matches the E-045 line). Milestone head ships at step 600; first curve point
at step 2000 (log_every); watchdogs armed (90 min stall / 40 h cap).

*(iv — 23:20Z)* **Node 3 was itself host-starved and was deleted at ~$9; node
4 carries the fixed input pipeline and is FAST.** Node 3 shipped four
10-minute cycles without reaching step 600 — >4.2 s/step, the chips idle
~95%: measured on equivalent shapes, the host pipeline at K=144 costs ~15
s/step of single-threaded numpy `standard_normal` (428M draws) plus ~2.2 s of
gather+cast against ~0.3 s of TPU step. Fix (commit
[c4ce2da](https://github.com/blauewelt/earth/commit/c4ce2da36), all
flag-gated, default-off, bit-stability proven by the new
`tests/test_jaxport_pipeline.py` P1–P3 + all 5 G5 gates green): noise drawn
ON DEVICE in the jitted step (`apply_znoise_jax` — same semantics, jax RNG
stream, recorded in the config line), fp16 transfer with exact on-device
cast, 8-thread gather, depth-2 prefetch; the launcher defaults all four on
and folds in the tmpfs disk-guard fallback. **Node 4 (boot 22:35Z, training
≈22:57Z): step 2,000 at wall 367.5 s and 2,000→4,000 in 338.1 s =**
**0.169 s/step — 25× node 3, 7.2× #478's H100 torch pace.** Health textbook:
zmse 3.92→2.65, val_zmse 4.82→3.31, amp 0.89→0.94, clip_frac 0.024 in the
warmup window then 0, nonfinite 0, grad_norm_max 320.8 absorbed by clip 128.
**Projection: 200k ≈ 9.4 h ≈ $90 train leg (+~$14 of dead nodes) — inside
the pre-registered decision threshold; the run is GO. 200k ETA ≈08:30Z
08-26.** First rolling checkpoint (2.4G) confirmed on the bucket.

**05:55Z 08-26 — E-051 MID-RUN: OPERATOR APPROVES ROLL-FIRST; ETA CORRECTED
FOR THE PROBE/EVAL TAX.** TL;DR — the run is healthy and still learning at
40% of budget; the plan for the finished head is now fixed: roll first, then
decide about more steps. *(i)* Step 80,000/200,000, val_zmse **0.97119**
(ratio-equivalent 0.0453 on its monitor persistence 21.446) and still
falling: 1.181 @50k → 0.971 @80k. Train 0.753, amp healthy, clean pace
holding **0.166 s/step**. *(ii)* ETA CORRECTION, on the record: the $90/8:30Z
projection priced CLEAN steps only; the measured ALL-IN pace is **0.275
s/step** (monitor evals every 2k + the in-training probe every 20k are the
difference — 22,025 s of wall at step 80k, not 13,520). **200k now lands
≈13:50–14:15Z, train leg ≈$155** — still inside the pre-registered ≤$250 GO
band. *(iii)* **OPERATOR APPROVAL (Chris, 08-26: "Agreed, please go
ahead.")**: on the 200k head, run the **ROLL FIRST** (E-044b-roll protocol:
stage `sroll`, horizon 73, 3 starts/holdout-year, ONLY `horizon_auc_daymatched`
on `corridor_trainlon` comparable, vs the monthly pair's 0.939 and pentad
−0.499; gate uncertified at pentad by construction — first reading), THEN
bring the continuation decision (resume from the bucket npz is a true
continuation, ≈$50/100k steps; LR will be ≈3e-5 at 200k, so a long-halflife
fresh dispatch is the cleaner instrument if the curve demands a bigger run).
Continuation itself is NOT pre-approved — it returns to Chris with the
final curve slope. The head's torch export ships with the finals; publish it
to model-checkpoints-v1 before the sroll dispatch names it.

**00:05Z 08-26 — LADDER CLOSE-OUT: #484 CANCELLED ON THE SAME CPU SIGNATURE AS
#476 — E-045.3 IS BLOCKED, AND THE MECHANISM IS NOW A/B-CONFIRMED AS
CONFIG-TIED, NOT BOX-TIED.**

TL;DR — the K=48 rung's trainer fell off the GPU twice in a row, once on
either side of a full box stop/start, so the fault travels with the strided
configuration, not with the machine; the rung is parked until the
slice-the-published-Z fix exists, and the factorial's verdict stands complete
without it. *(a)* **#484 (E-045.3 3rd copy) CANCELLED 23:53Z** at ~57 min into
stage 2 with ZERO `stage2_step` records and the box frame reading gpu 0% /
cpu 96% on two samples eight minutes apart — the exact #476 signature
(11.3–12.5 s/step, gpu 0/temp 0/cpu 104). Its embed leg was HEALTHY (1,047
strided bins at 7.755→9.8 s/bin, ≈$5.5, GPU pace throughout) — same shape as
#476: GPU embed, CPU trainer. **The A/B the re-dispatch was designed for is
answered: box 48632885 was stop/started between the two copies, so a wedged
CUDA state is excluded; the strided stage-2 path itself loses the device.**
Cost of #484: ≈$7.5. *(b)* **E-045.3 is recorded BLOCKED** — 2 copies, ~$18,
zero step records; do not dispatch a 3rd copy of the same configuration
(§4.13 in advance). The unblocking affordance is the registered
**slice-the-published-Z** proposal: read the complete published
`Z_8b639abe36_37e146384b` (3142,P,d_z), take rows `ts`, write a NORMAL cached
Z the trainer memmaps — no strided embed, no RAM-held Z, ≈$4.5-per-rung embed
tax also deleted. Chris's call to build or park (E-051's host-starved node 3
is an independent second instance of a host-bound trainer beside a healthy
device — the shape is real on two tiers). *(c)* **The factorial is CLOSED
without the 15-day interpolating rung**: span-restored 30 d 0.0713 · 10 d
0.0804 · 5 d 0.0820 (#478) — flat, against the K-fixed 0.07→0.51 cliff.
Figure `ml/figures/fig_factorial2.png` regenerated; paper updated (abstract's
"intrinsic to the step size" reading retired in favour of the factorial's
span story, sec:factorial completed with the landed 0.0820, fig_factorial
re-made with the decisive point, tab:ablations rows resolved). *(d)* Box
48632885 queue-checked and released after cancel (no queued run pinned to
it). #483 confirmed IN PROGRESS on the H100 since #478's 23:33:26Z drain
(codec-probes phase; its month-block re-embed and the header_t-516 §5.26
publish watch belong to the overnight checks).

**00:50Z 08-26 — NIGHT CHECK: #478'S ARCHIVE IS REAL (NOT GREEN-AND-VOID),
AND #483 IS RE-EMBEDDING BECAUSE THE BLOCK Z WAS NEVER ON THE CACHE — THE
"~$2 RE-RUN" EXPECTATION RESTED ON A MISATTRIBUTED CACHE KEY.**

*(a) #478 (E-045.1, the decisive span rung) drained GREEN with a real verdict
bundle* — `probes-478.json` (384 KB), `run-478.jsonl` and `plan-478.json` all
on `ml-metrics`; the 0.0820 harvest below stands on an archived artefact, not
a live branch. *(b) #483 (E-047-HEAD 6th, fusion-vs-selection) picked up on
the H100 at 23:33Z — but it is RE-EMBEDDING the month-block z from scratch*
(516 blocks at ~47 s/block, embed ends ≈06:15Z, head after; verdict ≈midday)
*rather than pulling a cache, and the measured reason is that NO
embed-cache-v1 asset exists under the block codec's key.* Both complete
12-chunk Zs on the release — `Z_071ef7e181_37e146384b` (published 06:27Z from
#463's pass) and `Z_8b639abe36_37e146384b` (published 10:37–11:07Z during
#470's window) — are byte-identical in shape: 17,433,927,552 bytes =
(3142 × 86698 × 32) float16 + npy header, i.e. per-bin d_z-32 CONTINUOUS Zs
(and E-051's launcher verified 8b639abe36 against the run-415 identity by
VALUE, z_rms 5.17825). The 13:20Z 08-25 block's claim that 8b639abe36 was
"the e047a-tpu-60k block codec × pentad tensor" is therefore WITHDRAWN — two
keys, one substrate; the second key most plausibly reflects a re-uploaded
run-415 checkpoint hashing differently — and with it the "the 6th dispatch
pulls Z in minutes … ~$2" estimate: #483 pays its own block embed (~$15 on
the H100). The run still answers its question; nothing needs cancelling.
Watch its own `job_timeout` at the morning check — embed+head ≈ 12–13 h.
*(c) #480 (E-049a continuous d_z-6 control) healthy at step 150,000/200,000*
(probe temporal_r_deseas 0.55, chan_mse 0.683 vs persistence 0.843); lands
≈04:30Z + finals. E-050 stays gated on it.

**23:10Z 08-25 — EVENING HARVEST: THE DECISIVE RUNG LANDS AT 0.0820 — THE
CONTEXT STORY IS CONFIRMED AT THE HARDEST STEP SIZE.**

*(a) #478 (E-045.1, 5-day steps, K=144 = 720-d span — the decisive rung) FINAL
20k one-step ratio **1.7418 / 21.2544 = 0.0820**, deep inside the pre-registered
context band (~0.07–0.15; the step story predicted ~0.5).* The E-045 factorial
is now closed at its hardest cell: with span fixed at two years, 5-day steps
score 0.0820 where the K-fixed ladder scored the SAME step size at 0.505 —
span-restored rungs read 30 d 0.072 (A2a) · 10 d 0.0804 (#462) · **5 d 0.0820
(#478)**, a flat line against the K-fixed ladder's 0.072 → 0.505 cliff. Step
size does almost nothing; context span does almost everything, all the way
down. Secondary read-outs, noted not registered: `rapid_r_kfold` 0.679
[0.62, 0.729], deseas 0.585 at 20k (the pooled light-probe's contrary drift
recorded at 19:00Z stayed contrary and stays excluded by protocol). Scale
block source-written (206,658,592 params · batch 256 · 20,000 steps ·
240,933,742 windows). n=1 per rung (§3b) — the RUNG PATTERN is the finding.
This also hands **E-051** (the JAX full-budget run at this exact geometry) its
torch 20k certificate: E-051's registered comparator is now a measured 0.0820,
not an in-flight number. Job still `in_progress` at 22:55Z (probe ladder +
publish); **#483 (E-047-HEAD, 6th) remains queued behind it** on the H100 and
should pick up on drain — verify it pulls `Z_8b639abe36` from the cache.

*(b) #484 (E-045.3, K=48, 3rd copy) is past the first-minutes check that
killed #476: embed ran at GPU pace (~9.3 s/bin, 1047 bins ≈ 2.6 h, vs #476's
11.27 s/STEP CPU signature), stage-2 config written (K=48, 206.56M,
80,282,348 windows), monitor line healthy (val_persistence 26.804, znoise
rel 0.135 — pentad regime). First curve point at step 2000 pending; 20k lands
~early morning 08-26. Completes the span-fixed ladder's last rung.*

*(c) #480 (E-049a continuous d_z-6 control, E-050's warm-start parent) is
healthy at step 123,000/200,000 — 0.239 s/step, loss_rec 0.237, in-training
probes steady (temporal_r_deseas ~0.55, chan_mse 0.677 vs persistence 0.843).
Lands ~04–06Z 08-26 plus finals. E-050's first dispatch stays GATED on it;
recipe + plan doc still owed before that dispatch.*

*(d) Owed, unchanged: #472/#473 Tier-1 recon vs their TPU siblings (the
cross-framework certificates); #482's corpse diagnosis on box 48478309
(pre-LN vs post-LN constancy decides E-050's bound); E-046's seed-0 pair.*

**19:00–19:20Z 08-25 — EVENING STEP: THE K=48 RUNG'S TRAINER RAN ON CPU AND WAS
KILLED AT STEP 600; THE DECISIVE RUNG IS DEEP IN ITS PREDICTED BAND.**

*(a) #476 (E-045.3, K=48 half-month rung, 2nd copy) CANCELLED at step 600/20,000 —
11.27 s/step, the CPU signature.* Its strided embed ran at GPU pace (7.4 s/bin, ~2 h,
correct-by-design re-embed: a strided rung's cache is disabled on purpose since #462's
poisoning), then stage-2 crawled at 9× the SLOWER-than-K=144 pace with the box frame
reading gpu 0%/temp 0/cpu 104% — the trainer lost the device between phases, or the
GPU died after #473 drained. At that pace 20k = ~63 h against the 900-min timeout: the
run could not answer its question (§4.13). Box 48632885 stop/started to clear any
wedged CUDA state; **#484 (E-045.3, 3rd copy, same H100)** dispatched with the
first-minutes check written into its doc: embed at ~7–8 s/bin = healthy, CPU-class bin
times = a dead GPU, destroy and re-rent, never debug a lemon. Cost of #476: ~$10 for
an embed a strided run cannot keep.

*(b) #478 (E-045.1, K=144 — the decisive rung) at step 15,400: in-flight ratio
0.0905, DEEP in the registered 0.07–0.15 context band and still falling*, clip_frac 0
throughout, pace holding 1.21 s/step; 20k ≈21:55Z. The pooled light-probe series
(0.55–0.63, drifting down while val falls) disagrees in direction — recorded, and
excluded by protocol; the registered verdict is the one-step ratio.

*(c) #472 (E-047b, the 126.9M Chinchilla-anchored month-block codec) finals, from the
archive: 200,000 steps reached, final heavy probe z_mse 10.625/19.891 = pooled-legacy
0.534, linear_r_deseas 0.309.* Its probe bundle carries THREE files, not eight — the
month-block probe gap (probe_kfold/dip_check/probe_head never pass the block map,
probe_sequence dies on a 3-vs-2 unpack) now confirmed on a second run after #473;
mechanism and line numbers in `claude/expectations.md`. The codec itself survives as
the 30-day artifact `pixelmae-472` (expires 2026-09-24 — publish to the release or
write off before then). Its box was stopped on idle burn at 15:21Z by the hourly
watch.

**14:30–15:00Z 08-25 — THE E-046 VERDICT LANDS ON ITS STRONGEST BRANCH: THE FSQ
CODEC'S HEAD READS 0.4394, BELOW BOTH CONTROLS — AND THE SEED PAIR AGREES TO 1e-4
WHILE DISAGREEING COMPLETELY ABOUT SURVIVING THE TRAINING.**

*(a) #477 (E-046-HEAD seed 1, the FSQ verdict head on run-455's lattice z) GREEN at
14:26Z, full ladder archived (`probes-477.json`).* **The registered verdict: 20k
one-step ratio 0.6497/1.4788 = 0.4394 — below A9 (#446, read-time 8-level quant) at
0.4916 and well below the continuous pair #427/#432 at 0.5056/0.5045.** That is the
pre-registered "at-or-below A9" branch: training THROUGH the lattice beats
quantizing at read time, which beats not quantizing — the FSQ codec earns its place
at pentad, and the resolution story (A9's discovery) survives its strongest test.
Two riders, both measured. (1) The number is accidentally a seed pair: #468 (seed
0)'s last healthy reading was 0.4395 at step 15,600, matching #477's 20k 0.4394 to
1e-4 — though at a different step, so it corroborates rather than replicates. (2)
The z-space is violently spiky to train on: seed 1 logged EIGHT gmax>10,000
excursions (20,077 · 39,200 · 101,323 · **788,370** — larger than seed 0's worst —
· 15,278 · 75,473 · 183,661 · 52,612), every one absorbed by grad-clip 128 and the
run completed, where seed 0 died of its seventh and each continuous control logged
ONE nonzero clip in 200k steps. So #468's death was draw-luck inside a regime both
seeds share, the E-044b clip-as-mechanism claim gains its hardest instances yet,
and the FSQ forecast win is priced: an optimization landscape that can kill a run.
Folded into the paper (sec:quant + tab:ablations E-046 row + tab:inflight). The
pooled probe series (~0.44–0.52) stays excluded by protocol. **FOLLOW-UPS EXECUTED
on Chris's approval (~15:30Z, "Proposal sounds good, let's do it!"):** (i) #468
(E-046-HEAD seed 0)'s box-local ~15,600-step head is WRITTEN OFF — its 20k twin's
full ladder is archived as `probes-477.json`, so the partial adds nothing; Vast
48520137 (gpu-box-32966687) leaves the STOP-ONLY / NEVER-DESTROY list (it currently
runs #480, the E-049a d_z-6 control). (ii) Warm-start quantization is PROMOTED ahead
of any further cold-start FSQ arm, and its affordance is landed as **E-050 /
`--fsq-warmstart`** (6c83ed3): resume a CONTINUOUS checkpoint with the lattice
switched on — a four-key hole in the resume guard, parameter-free so every weight
loads exactly, `--fsq-auto-step` rebased onto the resume step, provenance recorded
as `fsq_warmstart_from`, recipe-only workflow key, 7 new guard tests. The first
warm-start dispatch is gated on #480 (E-049a, the continuous d_z-6 parent) finishing
~tomorrow noon. This lands ON TOP of
E-049's cold-start collapses (#481/#482) and sharpens the fork: run-455's z — a
DEGENERATED sign code — still forecasts better than the continuous codec, so the
lattice is worth stabilizing, which is what E-049b's intrinsic bound and the
warm-start-quantization repair both aim at.

*(b) Codec finishes: #473 (E-047a-fast, the torch twin of e047a-tpu, 45k→60k)
GREEN ~14:10Z, probes archived (`probes-473.json`) — the cross-framework Tier-1
audit (e047a-tpu vs #473, same recipe, two frameworks) is now RUNNABLE; named next
step, not started unprompted. #472 (E-047b, the 126.9M month-block codec) reached
200k and is in its final probe set (heavy probe: pooled-legacy z_vs_persistence
10.625/19.891 = 0.534); harvest on drain, then its box frees.*

**13:20Z 08-25 — #470 (E-047-HEAD, 5th dispatch, the fusion-vs-selection verdict head)
WENT GREEN AND IS VOID: stage-2 CUDA OOM, and the run's colour lied again.** The
trainer died at 12:01Z trying to allocate **13.22 GiB on the 24 GB card**
(`torch.OutOfMemoryError`, run backgrounded → job green; the §7 signature, found by
the archive's file list — `probes-470.json` carries provenance.json ONLY, no
temporal.json). Mechanism: the month-block z is **d_z 64**, so the xl144-zn head's
input doubles against every pentad head that fits this card at d_z 32; #461 already
measured the same class at K=48 (its fix was the H100). **The silver lining is this
morning's partial-publish machinery working as designed:** #470's embed pass
completed and published `Z_8b639abe36_37e146384b` (12 chunks, the e047a-tpu-60k
block codec × pentad tensor) to `embed-cache-v1` before the trainer died — so the
6th dispatch pulls Z in minutes and pays only the ~1–2 h head training, on an H100.
**#483 (E-047-HEAD, 6th dispatch, H100)** queued on `gpu-box-48254133` behind #478.

**08:00–10:00Z 08-25 — MORNING TRIAGE: THE CLEAN PENTAD Z IS PUBLISHED, TWO HEADS DIE
IN ONE MORNING (one silently, one loudly), AND ALL THREE ARE BACK ON THE BOARD WITHIN
AN HOUR — AT PULL-THE-CACHE PRICES, NOT RE-EMBED PRICES.**

*(a) The clean full pentad Z is DURABLE: `Z_071ef7e181_37e146384b` (run-415 codec ×
family4_na025_pentad_r2), 12 chunks (11 × 1.46 GiB + 0.12), landed on `embed-cache-v1`
06:27–06:37Z from #463 (E-045.1 K=144 rung)'s embed pass* — onto the key the poisoned
chimera was deleted from the day before. Every remaining E-045 rung now pulls it in
minutes; the three-times-embedded week is over. In the same window the partial-publish
machinery shipped (2ab7e83: chunks publish DURING the embed every ~10 min with a
manifest, an interrupted pass resumes at the first missing row on any box; 0af27a2: the
pull loop's 64-chunk cap — which a daily-cadence Z at ~78 chunks would have silently
truncated into the exact chimera `verify()` rejects — raised to the naming's own 676
ceiling), and two standing rules were written into `ml/CLAUDE.md` §5.26/§5.27 (partial
progress to a SAFE location; a standing-expectations ledger, now live as the project doc
`claude/expectations.md`, diffed at every check-in).

*(b) #463 (E-045.1, the DECISIVE span rung) — its trainer died SILENTLY at step
4,000/20,000.* Metrics froze at ~08:10Z (step 4000, val 3.409, grads healthy — 6.03,
clip_frac 0), GPU idle at 4% for 75+ min while the live publisher kept force-pushing the
unchanged file: the #196 backgrounded-trainer signature, caught by the §5.27-style diff
rather than by any run colour. Cancelled 09:37Z. Nothing scientific lost — the embed had
already published (a). **#475 (E-045.1 re-dispatch, first copy)** was cancelled while
still queued for carrying the inherited `job_timeout` 900 into a job the fleet check had
just priced at ~810–830 min + ladder (the #438/#450/#453 kill bracket);
**#478 (E-045.1 re-run, K=144, `job_timeout` 1400, gpu-box-48254133)** replaces it —
which then sat 9 min queued against the online+idle runner (the known post-cancel queue
stall; cancel + rerun cleared it in 90 s, the remedy's third confirmation). Pulls the
published Z, trains the same head from step 0 on the intact expdecay schedule; 20k
≈22:00Z, drain ≈23:30Z–00:00Z.

*(c) #468 (E-046-HEAD, the FSQ verdict head on run-455's lattice z) DIVERGED FATALLY at
step 15,800/20,000 and was cancelled at 08:30Z on its own registered falsifier* — the
SEVENTH gmax>10,000 excursion (323,120 against clip 128), loss 0.559 → 1,950 in one
logging interval, amp 0.92 → 20.8, and the trainer died with it. Against the continuous
controls #427/#432 (ONE nonzero clip_frac in 200k steps each), the FSQ z-space is a
different training regime — that fact is itself E-046/E-048 evidence. **The cut-off
reading, stated with its caveat: at step 15,600 the head read val 0.65525 / persistence
1.49105 = ratio 0.4395 — BELOW both pre-registered controls (#427 continuous 0.5036 at
20k in-flight; A9 read-time-quant 0.4916 final), i.e. trending toward "FSQ codec earns
its place at pentad" — but it is a step-15,600 number from a run that never finished,
and the registered verdict is the 20k one.** (The pooled in-training probe read the
OPPOSITE direction — ~0.45 flat vs the continuous 0.53–0.60 band — and stays excluded
by protocol; two instruments disagreeing is one more reason to finish the registered
one.) **#477 (E-046-HEAD re-run, SEED 1, gpu-box-32966687 restarted)** now runs it:
seed 0 would replay the same batch order into the same cliff, seed 1 changes only the
draw. Box-local Z reused; 20k ≈14:00–14:30Z. #468's ~15,600-step head survives on that
box's disk (rescue step ran on #477's start — verify the upload before striking the
sole-copy line in `claude/expectations.md`).

*(d) #476 (E-045.3, K=48 half-month rung — held by Chris pending a clean Z) RE-QUEUED*
behind #473 (E-047a-fast torch twin finish) on the second H100; starts ≈13:00–13:30Z,
pulls the published Z. The span-fixed ladder is whole again: #462 (K=72) 0.0804 landed,
#478 (K=144) and #476 (K=48) in flight, A8/A11 anchor the K-fixed row.

**00:30–01:00Z 08-25 — NIGHT STEP: THE SECOND SPAN-FIXED RUNG CONFIRMS THE CONTEXT
STORY, #456 IS RESCUED PRE-EMPTIVELY, AND THE SCALE FIT CLOSES THE COLLAPSE MODE BUT
NOT THE DRIFT.**

*(g) E-045.2 LANDS AT 0.0804 — the context story's second confirmation.* **#462 (E-045.2,
10-day steps, K=72, span fixed at 720 d) final 20k ratio 2.1109/26.2455 = 0.0804.** The
K-fixed ladder scored 10-day steps at 0.3377 (A8, #452 — K=24, span 240 d); with the SAME
step size and the span restored to two years the task drops to 0.080 — A2a-class
(0.0713), a 4.2× change from context alone. The factorial now reads: step size does
almost nothing (0.506→0.527 at 120 d span; 0.071→0.080 at 720 d span), span does almost
everything. **#463 (E-045.1, 5-day steps, K=144 — the decisive rung, prediction on
record: ~0.07–0.15 context story / ~0.5 step story)** is next on the H100. n=1 per rung
(§3b); the RUNG PATTERN is the finding, not any single level.

*(h) #456 (E-047b, the 126.943M Chinchilla-anchored month-block codec) RESCUED BEFORE its
timeout, not after.* Cancelled deliberately at 00:33Z at step ~124k — its 1400-min
job_timeout could never hold 200k steps at ~100 steps/min, the fourth of the timeout-kill
class (#438/#450/#453) but the first pre-empted: cancelling on a fresh token guarantees
the milestone artifact uploads (the #419 24-h-token lesson). Milestone verified OFF THE
FILE (tag run-456, step 120000, opt+sched PRESENT — a true resume, cosine schedule
restores at the same T_max) and published as `run-456__pixelmae.pt`; **#472 (E-047b
FINISH, remaining 80k steps, same box)** dispatched — after #471, its first copy, sat
11 min queued against an online idle runner (the known queue-stall; cancel + re-dispatch
cleared it in 60 s). ~40 min of training (120k→124k) is the rescue's total cost.

*(i) e048a2 AT MID-RUN: the scale fit CLOSED the constant-collapse mode and EXPOSED the
drift as the remaining disease.* At step ~28k of 60k: probe r alive (0.262 at 5k — where
the collapsed run read exactly 0.000 — then 0.06–0.15 noisy), collapse guard quiet,
sat_frac 0.10–0.50, encoder input-dependent. But `fsq_prequant.std_med` runs
0.73 → 3.6 → 5.0 → 8.4 → 20 across the run — the pre-quant scale still has NO restoring
force, the step-2000 lattice is outgrown by step 15k (sat_frac climbing again), and
loss_rec sits at ~0.327, the ctx-floor, where the CONTINUOUS month-block codec (e047a,
near-identical windowing) reached 0.098. READING: the windowing is fine (e047a proves
it); the lattice + unbounded to_z is what costs reconstruction. Arm B (w6s3) stays HELD —
same recipe, same disease. THE DESIGN FORK FOR THE MORNING, stated before the run ends:
either give the latent an intrinsic bound (normalization before the lattice / weight
decay on to_z) so the fit has a stationary target, or read E-046's #468 verdict first —
if lattice z forecasts no better than continuous even at pentad, the FSQ codec line
closes on its own. e048a2 runs to 60k for the complete trace (~$5, self-reaps ~03:30Z).

**20:30–22:00Z 08-24 — EVENING TRIAGE: two dispatch failures re-dispatched, the E-046
verdict run launched, AND E-048a's COLLAPSE DIAGNOSED — with a discovery about the
"healthy" FSQ codec on the way.**

*(a) #461 (E-045.3, K=48 span-fixed rung) DIED OF CUDA OOM AND WENT GREEN.* The recipe's
"the 24 GB class fits only K≤48" was arithmetic, not measurement: at stage-2 batch 256 the
K=48 window OOMed the HK 4090 at ~step 200 (3 GiB alloc, 2 GiB free, 21.5 GiB in use) and
the run concluded success anyway — the backgrounded-trainer signature (ml/CLAUDE.md §7),
caught because its archive stops at stage2_step 200. Re-dispatched as **#467 (E-045.3
re-run, 80 GB H100)**, queued behind #462 (E-045.2 K=72, healthy at 6.6k/20k, 0.82 s/step)
→ #463 (E-045.1 K=144, the decisive rung) → #465 (E-047a-fast finish of #453). Cost ~$0.3.

*(b) #460 (E-047-HEAD, fusion-vs-selection) REFUSED ON THE SAME d_z CONTRADICTION AS
#457 — with the top-level input correctly reading 64.* Mechanism, now understood: recipe
keys OVERRIDE dispatch inputs by construction (`${RECIPE_D_Z:-inputs.d_z}`), so the pentad
recipe's d_z 32 wins whatever the input says. A top-level input cannot fix a recipe; only
a recipe can — the month-block codec now has its own (`ml/recipes/xl144-zn-monthblock`,
codec fields read OFF the e047a-tpu-60k file, ac12672). Re-dispatched as **#466
(E-047-HEAD, 3rd dispatch, HK box)** — past the resume guard and embedding at 21:40Z.
Cost of the lesson: ~$0.4 across two refusals.

*(c) E-046 ENDGAME: the FSQ codec is done, its verdict head is running.* #455 (E-046 FSQ
pentad codec, 200k×512) finished at loss_rec 0.337; final full probe: pooled ridge NaN on
lattice z (the instrument gap pre-registered at dispatch — the collapse guard read NO
READING all run), mini z-ratio 0.6516/1.40625 = 0.463 in LATTICE units (not comparable to
run-415's continuous scale). Checkpoint verified off the file (tag run-455, step 200000,
d_z 32, fsq_levels 8, 512×12) and published as `run-455__pixelmae.pt`. **#468 (E-046-HEAD,
#427's exact 20k stage-2 window on run-455's z, Michigan box)** is the pre-registered
verdict; controls at step 20k: #427 (continuous) 10.704/21.254 = **0.5036** in-flight, A9
(#446, continuous + read-time 8-level input-quant) **0.4916** final. At-or-below A9 = the
FSQ codec earns its place; #427-parity = A9's win was read-time denoising; clearly above =
the lattice cost information.

*(d) E-048a ABORTED ITSELF AT STEP 10k — THE COLLAPSE GUARD'S FIRST LIVE KILL, AND THE
DIAGNOSIS IS A CONSTANT ENCODER, NOT SATURATION-AT-INIT.* The TPU window codec
(w6s6, 71M, --fsq-levels 8 --fsq-ladder auto) hit `linear_r_deseas` exactly 0.000 on two
consecutive probes and self-killed; node self-reaped (~$2 total). Measured on the step-8k
export (torch side, synthetic z-scored inputs): the encoder is INPUT-INDEPENDENT — one
distinct z vector across 1,024 samples, |pre-quant| ≈ 87 whether the input is data, zeros
or all-unobserved — and loss_rec was flat at the ctx-only floor (~0.33) from step 300, so
the collapse happened in the first ~300 steps, long before the step-2000 ladder fit (whose
"+0.00%" and quantization MSE 12,776 were the tell: the fit measured a distribution the
lattice could not reach). A FRESH model at the same arch is healthy at init (1.6%
saturation) — the collapse DEVELOPED; it is not an initialization artefact. A paired CPU
toy (both backends, w6s6 + fsq8 + controls, matched flags) does NOT reproduce it — all
four arms learn, input-dependent, near-full alphabet; JAX↔torch export parity 6.0e-08 —
so this is a scale/data-regime phenomenon, not a toy-reproducible backend defect. **Arm B
(w6s3) is HELD** until the fix below is in.

*(d2, 21:00–22:30Z) #466 (E-047-HEAD, 3rd dispatch) CLEARED THE GUARD AND OOMed THE EMBED
PASS — the first block-codec embed ever attempted.* `embed_everything`'s batch 8192 was
sized for 42-token per-bin samples; a k_time-7 month block is 282 tokens and attention
memory grows as B·tokens², so the same batch is ~45× the memory (1.10 GiB alloc against
314 MiB free on the 24 GB HK card). Fixed at 5b3b0bb — the embed batch now holds
B·tokens² at the per-bin value (181 for k_time 7; per-bin codecs bit-unchanged) — and
re-dispatched as #469 — **which OOMed the same way from the other side (21:58Z): "eval-only"
stage 1 TRAINED.** train.py's cross-tensor eval branch promises "no training will occur
(checkpoint step ≥ --steps)" by reading ck['step'], but the step was only ADOPTED when the
checkpoint also carried an optimizer — and the TPU exports are {args, model, step} with no
torch optimizer moments by construction, so the warm-start branch restarted s at 0 and
trained a fresh b512 block codec into #448's OOM, wearing an eval-only dispatch's clothes.
Fixed at 7b1ce1d (a weights+step checkpoint warm-starts AT its recorded step — the counter
is provenance) and re-dispatched as **#470 (E-047-HEAD, 5th dispatch, HK box)**. Five
dispatches, four distinct one-field lessons, ~$0.8 total: the d_z default trap (#457), the
recipe-override trap (#460), the block-embed batch trap (#466), the optimizer-less
warm-start trap (#469). Also surfaced by #469's loud cross-tensor print: convert.py stamps
args['data'] from its template, so e047a-tpu-60k.pt CLAIMS family3_na025 while it trained
on family4_na025_pentad_r2 — holdout_lon is correct and nothing mis-chose tonight, but the
export's provenance lies; convert.py fix is an open follow-up.

*(f, 22:00–22:45Z) THE SCALE FIT IS IMPLEMENTED, GATED, AND BACK ON THE TPU.* a1de68f:
`fsq_ladder.fit_auto` now searches ladder × base × PER-DIM SCALE (candidates from the
sample — 2·std_d, p90/p99/max of |v_d| at k∈{0.5,1} — default always wins ties, floored so
a constant dim cannot make a zero lattice); the fit string carries the radius (`u:<R>` /
`e<base>:<R>`, legacy bare entries = default scale, so every archived checkpoint rebuilds
bit-identically — pinned by test); `--fsq-auto-step` is a comma list in both trainers;
every light probe now logs `fsq_prequant {std_med, sat_frac}` — the monitor that would
have shown e048a's collapse at step 300 instead of step 10,000. Gates: 13/13 FSQ-ladder
checks (cross-backend parity to 1e-5 at per-dim scales), G4 5/5, config guards 5/5. Toy
(1,500 steps, w6s6 fsq8 JAX): fitted radii TRACK a still-growing |v| (4.06→5.06→7.92
chasing 3.07→4.21→5.53), distinct codes 66→115, out-of-bound fraction 0.81→0.21 — and
loss_rec is WORSE on the toy (0.174→0.200): a wider bound trades step size for usage, and
which way that trades AT SCALE is what the relaunch measures. **e048a2-w6s6 (TPU
v5litepod-4, on-demand) launched 22:27Z** on a1de68f with `--fsq-auto-step 50,200,2000`,
fresh bucket prefix (the collapsed run's evidence kept); ~$5, lands ~03:00Z. Arm B (w6s3)
still HELD until e048a2 shows a healthy early read. Pre-registered readings: fsq_prequant
sat_frac staying well under 1 + probe r > 0.05 = the collapse mode is closed; a repeat
collapse WITH the monitor = the constant-encoder attractor is not a scale artefact and
the design conversation reopens (LayerNorm-class normalization before the lattice).

*(e) THE DISCOVERY: the "healthy" FSQ codec is a SIGN CODE.* The same measurement run on
run-455's own trained weights: pre-quant |v| ≈ 3×10⁴ — fully saturated, one hundred times
further past the tanh bound than the collapsed e048a — and its information travels as
SIGN FLIPS: 15/32 dims flip with the input, ~1 bit/dim (synthetic-input caveat: a lower
bound; real batches may use more levels). So the E-046 codec that scores z-ratio 0.463
is effectively a ~2-level code wearing an 8-level lattice, and even healthy toys run
pre-quant to 3–6× the bound. THE MECHANISM UNDER BOTH FINDINGS IS THE SAME: the codec-side
FSQ fixes its scale at σ=1 by design ("no Z to measure") and the fit fits only
exp-vs-uniform per dim, never the SCALE — Chris's original E-048 ask ("for each channel
compute a distribution such that the levels can be level·c or c^level") was only half
implemented. The half that was skipped is exactly the failure mode: an unfitted c leaves
the lattice where the distribution is not. NEXT (tonight): complete the fit — per-dim c
measured on real pre-quant values, fitted EARLY (step ~200, before the drift completes)
and refitted at 2000, recorded in the fit string so checkpoints rebuild exactly; archived
uniform checkpoints stay bit-identical. Then relaunch e048a; arm B follows a healthy
step-3000 read. n=1 everywhere here (§3b).

**18:40Z 08-24 — A11 OVERTURNS THE CADENCE READING: CONTEXT, NOT STEP SIZE, DROVE THE
LADDER.** **E-045-A11 (#464, 30-day steps with the context cut to the 5-day arm's 120 d —
stride 6, K=4): final 20k ratio 0.5274** (z_mse 15.0506 / persistence 28.5362). The
factorial square so far: (5 d step, 120 d) 0.506 · (30 d step, 720 d) 0.071 · **(30 d
step, 120 d) 0.527** — at MATCHED 120 d context the 30-day task is exactly as hard as the
5-day one, so the K-fixed ladder was tracking the CONTEXT AXIS, not the step. The
white-innovation-floor account is now WEAKENED where it was strong: A2a's model error
2.03 vs A11's 15.05 on the same 30 d targets says the "irreducible" content is largely
predictable GIVEN LONG CONTEXT. Leading mechanism candidate, pre-registered before the
span-fixed rungs land: THE SEASONAL ANALOG — at stride 6, K=24 reaches the same-calendar
frame one year back (12 strided frames), so the head can read last year's state; K=4
cannot, and neither can stride 1 at K=24. This is the calendar-replay machinery (E-043b)
showing up in one-step form. PREDICTIONS now on record: E-045.1 (5 d step, K=144, 720 d —
includes t−73, the year-back frame) lands ~0.07–0.15 if the context/analog story is
right, ~0.5 if step size is; E-045.3 (15 d, K=48) and E-045.2 (10 d, K=72) interpolate.
Caveat: A11 moved span AND K together (120 d ∧ 4 frames), so "context" here bundles
span with frame count — the span-fixed rungs unbundle it. n=1 (§3b). Chris's factorial
design is what caught this; the K-fixed ladder alone had the wrong story.

**18:00Z 08-24 — A10: THE ALPHABET WIN DOES NOT TRAVEL.** **E-045-A10 (#458, A7's
half-month configuration + `--input-quant 8`): final 20k ratio 0.1692** (z_mse 4.5391 /
persistence 26.8270) against A7's 0.1620 — +0.007, parity-to-slightly-worse, where the
same lattice bought −0.014 at the 5-day rung (A9's 0.4916 vs 0.5056/0.50447). READING:
input quantization helps exactly and only where the z is noise-dominated — consistent
with the white-innovation-floor account of the 5-day regime (denoising the input matters
when the input is mostly noise), and it sharpens E-046: the FSQ codec's case rests on the
pentad rung, and a coarser-cadence FSQ codec has no measured support. n=1 (§3b). The
in-training pooled probe read 0.567 at 20k, in band.

**14:30Z 08-24 — E-045.x, THE SPAN-FIXED LADDER (Chris's design, 14:15Z).** The original
ladder held K=24 frames and let context span shrink with the stride (720 d at 30 d steps
→ 120 d at 5 d steps) — step size and span were CONFOUNDED, which Chris caught. The new
ladder holds the span at TWO YEARS everywhere and feeds more frames as the step shrinks:
**E-045.3** = stride 3, `--K 48` (2× frames, predict +15 d) · **E-045.2** = stride 2,
`--K 72` (3×, +10 d) · **E-045.1** = stride 1, `--K 144` (6×, +5 d); the existing
A2a/b (stride 6, K=24, 720 d) is the completed corner, and A11 (K=4, 120 d span, 30 d
step) stays as the cheap fourth corner of the step × span factorial. HYPOTHESIS, either
way a result: if E-045.1 stays ~0.5-class with the full two years in view, short-step
difficulty is intrinsic (the white-innovation floor reading holds); if it drops well
below the K-fixed 0.506, the original ladder was substantially a CONTEXT-SPAN result and
the pentad head was starved of history, not of predictability. Head capacity is
DELIBERATELY FIXED (206M) — one variable; a capacity arm at the most interesting rung is
the pre-registered follow-up (A9's alphabet win already hints the head is input-capacity
bound, so K=144 at fixed params squeezes per-token capacity 6× and that is part of what
is measured). Mechanics: `--K` is a plain temporal.py flag (default 24; k_max follows K
for a fresh head), rides the sched tail. Sizing: memory ~∝K — .3 fits the 24 GB class;
.2/.1 go to the 80 GB H100 when #453 frees it (batch 256 kept). n=1 each (§3b).

**12:45Z 08-24 — THE CADENCE LADDER IS COMPLETE, THE ANCHOR-SIZED CODEC EXISTS, AND
THE UNPOOLED COLUMN STARTS FILLING.** Midday harvest:

- **E-045-A8 (#452, 10-day cadence rung) FINAL: 0.3377** (z_mse 8.8629 / persistence
  26.2455 at 20k). The ladder now reads, one codec, one head, only the step size moving:
  **30d 0.0721/0.0729 · 15d 0.1620 · 10d 0.3377 · 5d 0.5056/0.50447** — smooth,
  monotone, steeply super-linear in 1/Δt; no rung is a cliff, so no rung is a bug. n=1
  at the interior rungs (§3b).
- **E-047b-TPU LANDED: the 126.943M month-block codec**, 60k × 256 in ~4 h at 0.237
  s/step (node e047b-tpu-60k, self-reaped; published as `e047b-tpu-60k__pixelmae.pt`).
  First reads, pre-audit, n=1: final loss_rec 0.1428 (ABOVE the 40M twin's 0.0979 —
  half the samples at batch 256, and 60k is far under the anchor's appetite) while the
  in-training light probe reads BETTER all run (0.278 vs 0.242) — the wider z carries
  more state before recon converges. Tier-1 audit + 20k head decide.
- **E-043a-LADDER (#459) FILLS run-416's UNPOOLED DASH: 0.575 [0.445, 0.675]** vs its
  pooled ridge 0.613 [0.493, 0.716] — the unpooled head reads BELOW the pooled ridge on
  the monthly nolonhold codec (240 labels; overlapping CIs; a direction, not a level).
  Controls in the same bundle (raw-3x3, unpooled wind) for the paper's Table-4 row.
- **#457 (E-047-HEAD) FAILED CORRECTLY AND CHEAPLY**: the dispatch copied the A9
  template's `d_z 32` against the block codec's d_z 64 and the architecture-contradiction
  guard refused before training (~$0.2 of embed/transform). One field changed,
  re-dispatched (queues behind #458 on the HK box).
- **#450 (40M b128 long form) DIED TO ITS OWN job_timeout 900** at ~15 h, ~step 190k of
  200k — the timeout was copied from the fast-arm sizing against a 23 h run, the same
  §-class mistake as #438's. Training curve survives on ml-live-450; the last milestone
  checkpoint sits on box 48478309's disk; a resume-finish (~$1) is optional since the
  b512 twins (TPU done, #453 landing) carry the fast read. **#456 (126M b64) is in the
  same trap by construction** (job_timeout 1400 vs ~38 h needed, plus the 24 h token
  ceiling): decision tonight — resume-chain it or let the TPU 126M + audit carry the
  Chinchilla read.

**06:45Z 08-24 — THE FIRST MONTH-BLOCK CODEC EXISTS AND IS ACCEPTED; ITS HEAD AND TWO
FOLLOW-ONS DISPATCH NOW (Chris's morning direction).** The TPU 60k run (node
e047a-tpu-60k) finished all 60,000 steps in 2.8 h of training (0.167 s/step, loss_rec
0.098, light probe 0.242 at 60k) and self-reaped; the torch export is published as
release asset `e047a-tpu-60k__pixelmae.pt` (model-checkpoints-v1). **Tier-1 recon audit
(local CPU, 6 blocks × 512 px paired vs run-415's own recorded round-6 numbers, replay
check 0.05%): ACCEPTED.** The round-6 collapse is CURED at the anchor: fast channels at
the Argo cell read FVU 7.6%/17.4% (trained/held-out) where per-bin d_z-32 read 25–112%;
rg_t/rg_s at parity-or-better. The price is structural: Argo-free cells read 9–19%
where per-bin was <1% (one z per month is a real ~4:1 squeeze), and the WINDS carry it
— held-out winter tau FVU 0.65/0.75 even at full visibility, masked wind in-fill
overfits (0.30 trained vs >1 held-out at 0.69 epochs). Chinchilla reading at 40M/d_z 64:
OUTPUT capacity (decoder/d_z), not encoder scale, is the binding constraint; more data
or masking before more parameters at this rung. All n=1 at a new tier (JAX/TPU, §3b) —
#453, the same-recipe torch twin (~13:00Z), is the cross-framework check. Dispatching:

- **E-047-HEAD (block-z 20k stage-2, HK queue behind #452):** #427's exact head config
  (xl144-zn-pentad-nolonhold, stencil 145, znoise 0.7, grad-clip 128, seed 0, 20k) on
  the TPU block codec's z — the axis becomes MONTHLY by construction (temporal.py
  adopts k_time/time_block from the checkpoint; --time-stride refused on blocks).
  **Hypothesis: fusion beats selection — ratio below E-045-A2a's 0.0721** (selection,
  same tensor/head/steps); parity means one-bin-per-month suffices and the block codec
  buys the roll only. JAX-trained tier, never pooled.
- **E-045-A10 (does the quantization win generalize off-pentad?):** Chris's question —
  A9's gate fired at 5-day; is input-quant cadence-specific? A10 = A7's half-month
  configuration (stride 3, offset 2) + --input-quant 8. Control: A7's 0.1620.
  Hypothesis: below 0.1620 = the alphabet helps across cadences (an FSQ-15d codec
  becomes thinkable); parity = the win is pentad's regime. Third in the HK queue.
- **E-047b-TPU (the anchor-sized codec on the TPU, Chris: "spend is fine"):** same
  126.943M month-block geometry as #456 (1024×10, 8 heads, d_dec 768, d_z 128) at
  batch 256, steps 60k, node e047b-tpu-60k, on-demand (~$45 at a guessed ~0.5 s/step;
  memory arithmetic says batch 64/chip fits 16 GB HBM — the first step will say).
  #456 (torch b64 200k, Japan box) continues as the torch-tier long form.

<a id="e-046"></a>
**15:50Z 08-25 — E-046 RESOLVED AT ITS REGISTERED INSTRUMENT: THE FSQ CODEC EARNS ITS
PLACE AT PENTAD.** **#477 (E-046-HEAD seed 1 · #427's exact 206.5M xl144+znoise 20k
stage-2 window on run-455's lattice z · stage stage-2 · data family4_na025_pentad_r2 ·
resume z of run-455 · steps 20,000 × 256)** — final one-step ratio
**0.6497 / 1.4788 = 0.4394**, against the two registered bars: **A9 (read-time
input-quant) 0.4916** and **#427's own 20k point 0.5036**. Clearly below both — the
margin to A9 is ~46× the #427/#432 pair spread (0.00113). The at-or-below-A9 branch of
the registered reading fires: discretizing in the CODEC reproduces and exceeds the
head-side win, and is free downstream. Corroboration with its caveat: seed 0 (#468)
diverged fatally at 15.8k, but its cut-off reading at 15,600 was 0.4395 — near-identical,
though at a different step and unregistered. Nothing traded away on the instruments that
can see it: `rapid_r_kfold` 0.515 (pooled legacy), in-training probe 0.464. TWO CAVEATS
THAT TRAVEL WITH THE NUMBER: the ratio is dimensionless but lives in lattice-z space
(persistence hardness need not equal continuous-space hardness — pre-noted at #455's
finish), and the registered result is n=1 on a new codec tier (§3b: the winner buys its
pair; a seed-0 refinish must survive the gradient cliff that killed #468 — grad-clip or
seed luck). **The compound reading with the rest of the record is the striking part:**
run-455's z is measurably a ~32-bit SIGN CODE (the E-048 finding), and it still beats
the 1,024-bit continuous substrate by 0.065 as a forecasting substrate — the strongest
evidence yet for the capacity story, and for road B's premise that the restriction
itself helps. Road B's own d_z-6 arm fails at COLD-START OPTIMIZATION (#482, constant
encoder), not at information restriction: #480 (continuous d_z 6, same everything minus
the lattice+bound) is healthy past step 15k (loss_rec 0.249, step-0 probe persistence
z-MSE 0.4668 vs #482's collapsed 0.0009) — the natural experiment that convicts the
lattice-at-cold-start, and the reason the warm-start-quantization repair is next.

**02:30Z 08-24 — E-046 DISPATCHED (Chris's cost nod, 02:00Z: "the costs are acceptable").**
The FSQ bottleneck landed at 7f8dabb (`--fsq-levels`, default-off bit-identity pinned to
0e10253; InputQuant moved to model.py as the ONE quantizer; every codec loader rebuilds
the lattice or refuses; recipe-only key like time_block). **The arm is d_z 32 with L=8 on
all 32 dimensions (recipe f4r2-40M-fsq8) — deliberately NOT the paper's option-A d_z-5:**
the round-6 audit showed 40 channels already squeezed at 32 dims, and A9 measured the
[8]^32 alphabet WINNING head-side at 0.4916 — E-046 moves exactly that alphabet into the
codec, one variable. Hypothesis: the FSQ codec's 20k stage-2 ratio lands clearly below
#427's 20k point (read off ml-metrics run-427.jsonl at harvest, per the plan §6);
falsifier: at-or-above = continuity was not the binding constraint, answered for ~$9.
Steps 200,000 clean (NOT #415's max_minutes-refit accident of 197,428 — the codec-step
comparison is not the falsifier, the stage-2 20k point is). The plan §7 GPU smoke is
SKIPPED as answered: test 6 pins loader↔trainer encode bit-equality and test 8 trains
the lattice codec end to end on CPU. Fresh 258GB box (contract 48520137). Also
02:30Z: **#451's replacement dispatched** — same f4r2-126M-monthblock b64 200k config on
a fresh 128GB box (contract 48520140), since 48478310 has been Vast-offline 7+ h with
metrics frozen at step 9000; if the host resurrects, the FURTHER-ALONG of the two is
kept and the other cancelled (they are the same run re-attempted, not replicates — same
seed).

**01:30Z 08-24 — THE A9/E-046 GATE FIRES.** **E-045-A9 (#446, `--input-quant 8`, the
head's inputs snapped to an 8-level-per-dim alphabet): final 20k ratio 0.4916**
(z_mse 10.4486 / persistence 21.2544; rapid_r_kfold 0.677 [0.624, 0.725], in the
control band) against the 0.50560/0.50447 control pair — 0.0135 below the pair mean,
~12× the pair's own |Δ|. The pre-registered E-046 gate condition ("meaningfully below
the control") is met: **discretizing the pentad input helps one-step, so the E-046 FSQ
codec retrain is FUNDED** — dispatch waits on Chris's cost nod (morning summary), per
the overnight budget ambiguity. Ladder bookkeeping: A6 dead (0.5077), A9 is the first
arm to move the pentad number in the right direction. Ops, same check: **#451 (126M
long form) STALLED at step 9000 since ~19:15Z** — its box 48478310 is Vast-offline
while its runner still reads online+BUSY: frozen metrics beat runner status (tonight's
correction to the 22:30Z note); job_timeout ends it ~15:50Z if the host never returns;
morning decision (fresh re-dispatch vs a TPU 126M attempt) — #454's cancellation makes
this the only live 126M. And the first TPU 60k launch died at minute two to
unattended-upgrades holding the dpkg lock through all five 20 s retries (self-reap
worked; ~$0.30); fixed at cae981f (stop the service + DPkg::Lock::Timeout 600),
relaunched 01:40Z — revised codec ETA ~05:45Z. #453 re-measured at 1.04 s/step →
~13:45Z; #450 accelerated to ~0.21 s/step → lands ~08:45Z, BEFORE its own fast arm.

**NIGHT HARVEST, 23:00Z 08-23.** Four results and one cut:

- **E-045-A6 (#445, fine season phase) FINAL: 20k ratio 0.5077** (z_mse 10.791 /
  persistence 21.254; rapid_r_deseas 0.604) vs the 0.50560/0.50447 control pair —
  ~2.4 pair-spreads ABOVE the pair mean: the head-side continuous phase token does NOT
  improve pentad one-step. The staircase hypothesis is dead at the one-step tier; its
  roll-lock arm (does fine phase change calendar mode-locking?) still needs a roll and
  stays open. A9 (#446, input-quant — the E-046 gate) was at 18k with val_zmse 10.62
  (trending ≈0.500, marginally below the pair) at 22:35Z; verdict at its final.
- **THE TPU TRAINS THE BLOCK CODEC 4.5× FASTER THAN THE RENTED H100.** The JAX tier-3
  stage-1 trainer (commits f3978f5..d116055, parity gates G4a-e: loss parity 2.7e-07,
  one-step parity, round-trip .pt export) ran the full E-047a-fast geometry (40M
  month-block, batch 512) on a v5litepod-4: **0.206 s/step measured over a clean
  0→3000-step run** (2,490 block-samples/s; light probe 0.192 at 3k; two spot
  preemptions along the way, and the bucket-keyed resume worked exactly as designed —
  the third node found the finished state, added nothing, exported and reaped). The
  rented H100 SXM does 0.93 s/step on the same config (#453) — per-sample the TPU is
  ~8× the 4090 and ~4.5× the H100. **A full 60k-step TPU run (node e047a-tpu-60k,
  on-demand, ~$22, ETA ~03:15Z) launched 22:40Z** — §3b: a NEW TIER, never pooled with
  torch numbers; #453 is its exact same-recipe torch twin, which makes the pair the
  cross-framework validation. Ops lesson for the record: three "empty bucket" reads
  earlier tonight were a malformed access token whose 401 was misparsed as zero items —
  the measurement had in fact shipped everything. Assert the ERROR KEY of a JSON
  listing, not just the absence of items.
- **E-047b-fast (#454) CANCELLED at ~3.5k steps, box destroyed — a cost cut, not a
  failure.** Measured 1.58 s/step on the $2.37/h H100 NVL → 60k ≈ 27 h ≈ $63 to beat
  the $0.28/h long form #451 (Tue ~08:30Z) by ~7 h. Not defensible inside Chris's $100
  envelope; the 126M question rides on #451, and a TPU 126M attempt is the cheaper
  accelerant if wanted (per-chip memory arithmetic says batch 128/chip plausibly fits;
  unmeasured).
- **All four in-flight month-block runs share a dead FULL probe** (light probes fine):
  probe_now derived its axis length from the tensor's 3,142 bins against 516-row
  block-z — every eval_every probe died wrapped (`IndexError: 516`). Fixed at e424692
  (T from the embedded axis; block cells decoded each against its own source bin,
  rollout_spatial's rule; per-bin path proven byte-identical pre/post). In-flight runs
  keep the bug (non-fatal); every missing z%/chan% number is recomputable offline from
  their checkpoints on the fixed tree.

**FAST ARMS, dispatched ~20:00Z 08-23 (Chris: "$100, ideally the new codecs trained by
morning"): E-047a-fast + E-047b-fast — the SAME two codecs at their recipes' ORIGINAL
batches on hardware sized for them.** New recipes `f4r2-40M-monthblock-b512` (H100 SXM
80 GB, ~$1.49/h) and `f4r2-126M-monthblock-b256` (H100 NVL 94 GB, ~$1.47/h), steps
60,000 — the f3_anchor41M workhorse's own step-count class, and MORE data than the 200k
long forms (30.7M vs 25.6M block-samples at 40M; 15.4M vs 12.8M at 126M). Hypothesis
unchanged from E-047 (fusion beats selection; the fast arms exist to land a usable codec
overnight, ~9 h (a) / ~15 h (b) predicted from the measured 4090 step times × batch
scaling ÷ H100 class speedup). #450/#451 continue untouched as the long forms — same
family, two (batch, steps) points, reported side by side, never pooled (§3b). Estimated
~$15 (a-fast) + ~$23 (b-fast). Tier-1 recon audit + 20k heads + block-decode roll follow
on whichever codec lands first.

---

<a id="e-045"></a>
## E-045 · The pentad component ladder — six one-variable arms, DISPATCHING overnight 2026-08-22/23 (Chris's divide-and-conquer mandate, $20 envelope)

Question: which component breaks when the programme moves monthly → pentad? The ladder so
far, each rung measured 2026-08-22 evening at $0: the anomaly TRANSFORM reproduces the
recorded loss_rec (1.7 SE); the CODEC round-trips at FVU 0.4–0.6% on Argo-free bins
(healthy) and collapses on the fast channels only where 40 channels compete for d_z 32
(the 8% Argo-carrying bins); the PUBLISHED Z equals a local re-encode to one float16 ULP
(wrong-codec hypothesis dead — control: different bins differ by mean |Δ| 4.478). What
remains is the HEAD and the CADENCE, and the arms below split them. All use #427's exact
configuration (206.5M xl144+znoise head, frozen run-415 codec, grad-clip 128) at
temporal_steps 20,000 (#423 read 0.5404 at step 2,000, so a 20k monitor curve is a
readable one-step instrument), milestones 600 only, seed 0. Read-out: the stage2 monitor
ratio (val_zmse / val_persistence, each arm against ITS OWN persistence baseline) +
in-training probes. Bars: #427/#432 read ≈0.50 at 20k (pentad); the monthly xl tier reads
an order of magnitude lower. Instruments landed at 46d2a01 (`--time-stride/offset`,
`--target-bins-argo`, `--season-dropout`, `--season-phase`), all default-off with
bit-identity pinned in tests/test_e044c_knobs.py.

| arm | one variable | hypothesis (falsifier = the opposite branch) |
|---|---|---|
| **A2a** `--time-stride 6 --time-offset 2` | monthly CADENCE from pentad z, Argo-carrying bins (Chris's "pool 6 bins back into a month", by selection not averaging — z-means are off-manifold) | if the ratio lands in the monthly class, the pipeline is sound and pentad cadence/regime is the problem; pentad-class ratio here = something deeper |
| **A2b** `--time-stride 6 --time-offset 0` | monthly cadence, Argo-FREE bins (the clean 8-channel regime) | A2b good + A2a bad = the Argo-bin z regime is the poison; both good = switching/cadence; both bad = pipeline |
| **A3** `--target-bins-argo exclude` | pentad cadence, but never scored on regime-switch targets | if the grad-spike regime (#423's divergence class) disappears and the ratio improves, the regime SWITCHING is the mechanism that made clipping necessary |
| **A6** `--season-phase fine` | the season token staircase (all ~6 bins of a month share one token — Chris's catch, month_feats/ctx_all are integer-month sin/cos) is replaced by continuous fraction-of-year phase, head-side only | better one-step and/or changed mode-locking = the staircase mattered; unchanged = it did not |
| **A4** `--input-znoise 1.84` | the roll-repair lever: znoise rescaled to the pentad z-scale (0.7×2.63, restoring E-036/E-037's measured 0.3979× relative sigma) | one-step may WORSEN slightly (noise costs one-step, buys the roll) — judged by a later quick roll, not by this ratio alone |
| **A5** `--season-dropout 0.5` | Chris's anti-calendar-memorisation idea (note: no year token exists — the season token is month-only, and the "year" is carried by the state, which is the replay channel E-043b-PHASE identified) | judged by a later roll's mode-locking, not by one-step |

**INTERIM, 03:30Z 08-23 (chain step 2).** The 2x2 is resolving toward CADENCE:

- **A2a (#435, monthly cadence from pentad z, Argo bins): final ratio 0.0721**
  (z_mse 2.0339 / persistence 28.2068 at 20k) — **monthly-class**, on the same codec,
  same z, same head, same pixels as the 0.505 control; only the step size differs.
  The pentad z is a sound MONTHLY substrate even using the worst-recon-regime bins.
- **A3 (#438, pentad, Argo-target windows excluded): 0.570 at 18k with grad_norm 24**
  (control: ~0.50 and settled 2-3 by then) — the hypothesis INVERTS: the slow-density
  Argo targets were stabilising training, not poisoning it. Removing them hurts.
- **A2b (#437) failed on infrastructure**, twice removed from the science: a transient
  curl (56) reset killed the tensor pull (no retry — fixed in this commit, --retry 3),
  the fallback build then died writing the npz on a 43 GB-free disk. Retry pinned to
  gpu-box-31479844 (tensor already present there) after #435 drains.

**EXTENSION, 09:40–10:00Z (Chris's two directions, dispatched while he is away).**

- **A7 (#441, half-month cadence — stride 3 offset 2, ~15-day steps)** and **A8 (10-day
  cadence — stride 2 offset 0; queued behind A7 on the same box after 30257785's disk
  filled)**: the interior points of the CADENCE LADDER. Pre-registered reading: with
  stride 6 = 0.072/0.073 and stride 1 = 0.5056/0.50447 as the endpoints, a smooth
  monotone interpolation (≈0.15–0.2 at 15 d, ≈0.3 at 10 d) says the 5-day difficulty is
  INTRINSIC signal-to-noise; a cliff or non-monotonicity anywhere says some code path
  still assumes monthly cadence — Chris's suspected hidden bug, localised by which rung
  breaks.
- **A9 (--input-quant 8, pentad control + FSQ-style input tokenization, dc7cf13)**: the
  $0.6 version of Chris's capacity hypothesis — the head's input alphabet restricted to
  8 levels/dim (3 bits/dim) with straight-through gradients, targets continuous,
  spec+sigma riding the checkpoint so the roll honors it. Pre-registered: a one-step
  ratio meaningfully BELOW the 0.5056 control says input compression helps and funds the
  E-046 FSQ codec retrain (`ml/plans/E046_fsq_codec.md`, specced and priced ~$9 to a
  first verdict); parity or worse says the capacity story is weaker than it looks and
  E-046 waits. Takes the A5 slot on the HK chain (season-dropout defers — its readout
  needs a roll; Chris's active direction outranks it).

**LADDER FINALS + TWO OPS INSTANCES, 18:00Z (chain step 5).**

- **A7 (#441, stride 3 = 15-day cadence): ratio 0.1620** (4.3466 / 26.8270 at 20k; run
  shows `cancelled` — its OWN job_timeout 360 again, mid-ladder, training complete on
  ml-live). The cadence ladder now reads **30 d 0.072 · 15 d 0.162 · 5 d 0.505** —
  smooth and monotone through three rungs. No cliff, no hidden monthly assumption:
  the 5-day one-step difficulty is INTRINSIC. (In-training probes for the record:
  rapid_r_kfold 0.638 [0.579, 0.692] at stride 3 — labelled, pooled.)
- **A8 (#443, stride 2) is a GREEN-WITHOUT-THE-GOODS instance of the standing
  temporal.py blindspot**: run green at 73 min, metrics carrying ONLY config+resumed —
  the stage-2 head never trained (the pre-registered 51 GB-on-64 GB slice peak,
  OOM-killed behind the best-effort guards). Re-dispatched to the 515 GB box per the
  arm's own fallback line; the 10-day rung lands overnight.
- **The EIGHTH drain orphan on gpu-box-31479844** (identical signature, two live frames
  at 81.0 °C), stopped after queue check. Eight for eight on drains: the retirement case
  is now complete — recommend destroy/re-rent at the next natural break.

**FINALS, 09:40Z (chain step 3).**

- **A2b (#439, monthly cadence, Argo-FREE bins): 0.0729** (2.0638 / 28.3183) — twin to
  A2a's 0.0721. **The 2x2's monthly row is uniformly monthly-class**: at 30-day steps the
  pentad z is a sound substrate whichever representation regime supplies the states.
  CADENCE IS THE VARIABLE, now on both cells.
- **A3 final (#438, ml-live; run cancelled by its own job_timeout 360 at exactly 6h on
  the 2.2x box — training finished, ladder lost): ratio 0.5665**, grad_norm spiking to
  94.6 at 19k where the control settles at 2-3. Inversion final-grade: the Argo targets
  STABILISE pentad training.
- **A4 FINAL (#440, znoise 1.84): 0.8145** (17.3128 / 21.2544) — recovered from 1.051
  mid-run but still catastrophic against the 0.5056 control: the E-036/E-037 relative
  dose (0.3973x, confirmed on target in-run) is REFUTED as a transferable constant at
  pentad; any pentad exposure-bias noise needs its own calibration, and the window
  between "helps the roll" and "drowns the one-step signal" is far narrower here.
  (Interim record: at 14k it read 1.051 — worse than persistence,
  pre-clip grad_norm 215.8, with `input_znoise_rel_pers` 0.39732 confirming the rescale
  hit E-036/E-037's measured relative sigma exactly. The monthly exposure-bias dose is
  fatal at pentad: the 5-day z-dynamics are too weak-signal to survive noise sized on the
  monthly regime. The roll-repair lever must be sized DOWN from the pentad one-step
  error, not carried across as a relative constant. (Run left to finish for the record.)
- Fleet lore bought overnight: a refused Vast start ("state change queued") can execute
  HOURS later — check for surprise starts; the E-045 job_timeouts on the HK box needed
  600 min, not 360; 31479844 produced its SEVENTH drain orphan (same signature, verified
  two frames, stopped).

Reading so far: encode, transform, Z, head architecture and pipeline are all healthy;
**the 5-day step itself is the difficulty** — one-step ratio 0.505 vs 0.072 with cadence
as the only variable. The year-roll repair that follows from this is hierarchical: roll
at monthly stride THROUGH the pentad stack (A2a's head is exactly that object), keep the
pentad steps for within-month detail. A4/A6/A5 still to run.

Boxes: A2a/A2b on gpu-box-31479844 / gpu-box-46996216 (the stride slices X before
nan_to_num so the 85 GB peak shrinks ~6x and 64 GB boxes suffice; strided runs re-embed
their kept bins — 1/N of the pass — and deliberately publish nothing to the shared Z
cache); A3/A6/A4/A5 need the full-tensor peak and run SEQUENTIALLY on gpu-box-39184683
(515 GB RAM), chained by session wakeups after #432's drain. Cost: ~$2.2 + ~4 x $1.1
≈ **$6.5** of the $20. Rolls for the winning arms (short future-only srolls) follow
tomorrow on the day-matched protocol.

---

<a id="e-043b-phase"></a>
## E-043b-PHASE · Calendar or context? — RESOLVED 2026-08-22: **the pre-registered CALENDAR-REPLAY reading fires.** #434 (E-043b-PHASE, six-context-end hindcasts of the gate + the nolonhold s0 head) rolled 06:20→15:26Z on gpu-box-31479844

Dispatched 2026-08-22 06:20Z with the hypothesis pre-registered **in the dispatch doc**
(the log entry is written at harvest — a §1 debt, noted). The question: every head's
unforced future roll mode-locks to the calendar (gate 12-mo; the #414/#426 nolonhold pair
at exactly **36.0 months**, peaks Nov 2027/+36/+36, both seeds phase-identical, ACF 0.85 at
lag 36 and ~0 at 35/37), and from ONE context end that admits two readings: the phase is
the CALENDAR's (replay) or the phase is SELECTED BY THE STATE (a dynamical mode). #434
hindcasts both heads from six context ends — 2004-12, 2014-12, 2018-09, 2020-06, 2022-03,
2024-03 — via the new `longstart:` token (`long_multi`, f3ddcfb). Gate reproduced 0.643
EXACTLY → run VALID · params 211.4M head + frozen 40.7M f3 anchor · stage sroll ·
data family3_na025 · steps 0 (nothing trains) · cost ~9.1 h ≈ $2.7.

**RESULT.** The nolonhold head's post-record peaks are **pinned to the same calendar
months whatever the starting state**: from every one of the six context ends the smoothed
trajectory peaks at **Nov 2027, Nov 2030, Nov 2033, …** — six different initial states
spanning a decade, one attractor branch, calendar-locked. And the within-record halves
carry the second signature: rolls launched in 2014-12, 2018-09, 2020-06 and 2022-03 track
the subsequently observed record at **r_trained 0.70, 0.68, 0.70, 0.71** — *flat in lead
time*. A 10-year-old context "forecasts" 2023 exactly as well as a 15-month-old one.
Genuine forecast skill decays with lead; replay does not. Combined with the architecture
test (the 206M stencil head and the 1-pixel gate head correlate **0.93** with each other
over the 2004-context hindcast — 0.91 on held-out months — against 0.79/0.47 with
reality), the conclusion is:

**The 20-year hindcast tracking is trajectory replay — the training record is embedded in
the learned dynamics and the rolled state indexes it — and the archived `long` r's must
not be read as forecast skill.** This is E-021's finding (the 20-year fan's hindcast
skill was memorisation) reproduced at xl scale with a cleaner instrument. Scope guard on
the conclusion: it is about the LONG hindcast/future rolls. The 12-month scored blocks
(corridor AUC) are a separate instrument — truth-anchored starts, scored against held-out
years, with the #424/#425 controls behind them — but they too inherit a caveat this
finding sharpens: training windows may LOOK at held-out months as context
(`temporal.py:1474`, by documented design), so held-out-year skill on trained pixels is
softer than the words "held out" suggest. The gate head shows the same replay signatures
on its own 12-month lock (ac12 rising from 0.27 at the 2004 context to 0.87 at 2024-03).

**What would still rescue a dynamics reading, and how to test it:** a mode that is real
but weak would phase-lock to the calendar through the seasonal token while still carrying
state information in amplitude. The falsifier for THAT is ensemble dispersion under
context perturbation (E-021's under-dispersion test, rerunnable here), and it is not
dispatched — the replay reading already explains every observation with fewer parts.

---

<a id="e-044b-roll"></a>
## E-044b-roll · The pentad corridor AUC — RESOLVED 2026-08-22: **the hypothesis is REFUTED at n=1 — the pentad roll lands BELOW CLIMATOLOGY.** #433 (E-044b-roll, pentad sroll of #427's head, horizon 73 / starts 3 / dumproll) rolled 03:53→17:50Z on gpu-box-46996216

**RESULT (harvested 2026-08-22 ~19:30Z from `probes-433.json` on ml-metrics).** Day-matched
corridor AUC **−0.499** (corridor = corridor_trainlon, n_px 30,158 — no holdout, all
columns trained); window **−0.352**; gate scope **−0.365**. Raw 73-lead horizon_auc
−0.492, quoted only beside the day-matched form per §7b(g). Against the monthly nolonhold
pair's **+0.93933 / +0.93933** (#422 / #429), the pentad roll is not "worse by a band" —
it is **negative**: the rolled forecast's MSE exceeds climatology's by ~50% over the year.
The 20-year hindcast collapses the same way: r_trained **0.256**, r_heldout **0.016**,
lp18 **0.381**, amp **0.304** (the monthly heads read 0.79 / 0.47 / 0.88 / 0.74 on the
same protocol). Gate: the machine-readable SKIP (`certified: false`, pentad has no
reference) — the protocol behaved exactly as §7a specifies; this number is an UNCERTIFIED
FIRST READING at n=1, but the gap to refute (−1.44 AUC) is not a seed-band question.

**Reading.** Coherent with the one-step numbers, and quantitatively so: #427's one-step
ratio is 0.506 of persistence where the monthly arm's is 0.032 — ~16× the per-step relative
error — and the day-matched horizon takes **73** autoregressive steps where monthly takes
**12**. Six times as many steps, each injecting an order more error: the roll compounds it
into collapse. "6× cadence should improve things" is refuted in its strong form: the pentad
EMBEDDING may still be fine (the codec's k-fold numbers are ordinary), but the pentad
STAGE-2 HEAD cannot yet carry a year-long roll. The levers the result points at, in order:
the under-scaled znoise (this arm runs 0.1512× where E-036/E-037 measured 0.3979× — the
exposure-bias regulariser is 2.63× weaker than the one that bought +0.045/+0.050 at
monthly, exactly the mechanism a 73-step roll leans on), and per-step LR/schedule at pentad.

**MECHANISM, from the 2026-08-22 evening audit (channel autopsy of this roll + a local
copy-reconstruction audit of the run-415 codec — scripts and results in the session
record, `recon433/results.json`).** Four findings that recolour the number above:

1. **All 32 Argo T/S channels scored NaN in this roll** (`audit.per_channel_msss_clim_corridor`
   — only cur_speed, log_mld, ssh, sst and the four τ channels carry numbers). Cause,
   measured from the tensor itself: **`rg_*` is a monthly product written into exactly ONE
   pentad bin per month** (`n_rg_live` 252/3142 = 8.02%, deterministically the mid-month
   bin, offset 2), absent — not sparse — everywhere else. The −0.499 is an 8-fast-channel
   weather score; the monthly +0.939 averages 39 channels dominated by persistent deep
   density. **The two headline AUCs never measured the same quantity.**
2. **But channel composition does NOT explain the gap**: the monthly nolonhold heads
   restricted to the same fast channels still read **0.925/0.926** (recomputed from
   #422/#429's audits). Under E-043b-PHASE's replay finding, that monthly fast-channel
   0.93 is the replay mechanism working on trained pixels; the pentad head fails BOTH modes
   — its 20-year hindcast is flat near climatology (model sd 1.59 vs truth 3.96,
   r_trained 0.256), and its unforced future locks 1:1 annual (ACF peak at 73 steps =
   365 d; not 36 steps, so the K-window-echo mechanism is ruled out).
3. **The codec is HEALTHY — verified, not assumed**: a local encode→decode roundtrip with
   the exact training transform (verified by reproducing run-415's recorded loss_rec to
   1.7 SE; the roll itself also printed `Z cache verified vs live re-encode ✓`, so the
   right encoder ran) reads **FVU 0.4–0.6% (r 0.997)** on the 92% of bins that carry only
   the 8 fast channels — near-identity — and held-out ≤ trained everywhere (no
   memorisation in the codec).
4. **The real structural defect is the r2 tensor design meeting d_z 32**: on the 8% of
   bins that DO carry Argo, 40 channels compete for 32 latent dims and the fast channels'
   reconstruction **collapses** (cur_speed FVU 112%, ssh 86% — restored to 0.4–0.6% by
   hiding Argo from the encoder, same bins, same pixels). So the z-sequence the head must
   forecast SWITCHES REPRESENTATION REGIME every ~6th step, and the corridor scoring
   decodes through the lossy regime at exactly the rows where Argo truth exists. Deep-Argo
   FVU at those bins is ~2× the monthly d_z-64 anchor's (12–17% vs 6.9%), the direction
   and size that halving d_z while adding a channel predicts.

   Follow-ups this points at, in order: a pentad tensor revision that carries the monthly
   Argo state into every bin (persistence-fill or a separate slow-state pathway) so the
   representation regime stops oscillating; a d_z 64 pentad codec; then the znoise rescale
   for the head. Each is its own experiment with its own falsifier.

**Two mechanical notes.** (a) The run shows `completed failure` — that is step 22 ONLY:
`actions/upload-artifact@v4` refuses the dump filenames because the head label carries a
COLON (`roll_s145rspiral:111-…npz` — "path … is not valid"). The roll itself finished; the
JSON archived to ml-metrics (step 23 green). Fix: sanitize dump filenames. (b) The nine
dump trajectories (2.4 GB, 74/50/26 states × [86,698 × 32] f16 per start) exist ONLY on
gpu-box-46996216's stopped disk — stop-only, never destroy, until rescued or declared
expendable (given the collapse, the monthly nolonhold roll is the better animation
candidate anyway; a monthly `dumproll` re-roll costs ~3 h ≈ $1).

Written **at dispatch form, before the run exists** (§1: hypothesis first, so the log cannot
be rewritten to fit the answer). It is the second half of what Chris asked for on 2026-08-19
— *"report one-step MSE and corridor AUC"* — of which #427 delivered the first half only:
corridor AUC comes from a **separate `sroll:` dispatch** over the head #427 trained (spec
§7a/§7b). Code: `head_sha` **at dispatch**; the roll is `ml/rollout_spatial.py` at 52ad4a4.

**E-044b-roll · ROLLS #427's pentad xl144+znoise stage-2 head
(`head-weights-e044b-xl144zn-pentad-s0`) 365 DAYS forward over the pentad axis — horizon
**73 steps = 365.0 d**, day-matched to the monthly archive's 12 months (E-044 §7b), **3**
evenly-strided starts per holdout year (stride 24 pentads, phases ≈ 30 Dec / 29 Apr /
27 Aug), day-defined bands h1-18 / h19-36 / h37-73 — and DUMPS the full-window roll-forward
sequences for the UI's animation · params 206.5M head over a frozen 37.976M codec, **both
frozen, NOTHING trains** · stage `sroll` · data `family4_na025_pentad_r2` (C 40, T 3,142,
sha256 `37e146384b6f…`) · arch head 1024×16, K 24, stencil 145, ring
`spiral:111,4444,0.71,0.5`; codec 512×12, d_z 32, patch 1 · steps×batch **0 × 0** (the
`steps` field is the codec's own recorded 197,428) · resume `run-415`.**

[E-044 · the dispatch spec, §7a/§7b (project doc)](https://blauewelt.github.io/earth/docs.html?f=claude/E044-pentad-stage2-spec.md)

[E-044b · #427, the head this rolls](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-044b)

[E-043b · #422/#429, the monthly control pair](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043b-seed1-roll)

#### 1 · Hypothesis and falsifier

**HYPOTHESIS (Chris's, in his words: a 6× finer cadence "should improve things").** The
pentad head's **day-matched** corridor AUC — `horizon_auc_daymatched` on the
`corridor_trainlon` scope — is **at or above the monthly no-longitude-holdout pair's 0.939**.
The mechanism claimed is that a 5-day step lets the forecaster carry structure a 30-day step
integrates away, and that this survives all the way out to a 365-day horizon.

**FALSIFIER — and its weak half is stated first, because it is the honest part.** A
day-matched corridor `_trainlon` AUC **materially below 0.939** falsifies it. *"Materially"
cannot be quantified at this cadence today:* the pentad pair spread does not exist, because
no second pentad head has ever been rolled. The monthly xl `_trainlon` pair noise is
**|Δ| 0.00075 (#417) and 0.00108 (#418)**, pooled sd 0.00066 on 2 dof, with the xl tier's
blended 95% upper bound 0.0037 — those are the numbers a monthly comparison would be judged
against, and **they may not be borrowed across a cadence boundary** (§3b). So: a gap of
order 0.01 or larger is a result either way; a gap of order 0.001 is **uninterpretable until
the seed-1 pentad roll exists**, and must be written as "no measurable difference at n = 1",
never as agreement.

**CONTROL, named explicitly (§3).** The monthly no-longitude-holdout xl144 pair —
**#422 (seed 0) and #429 (seed 1)**, rolled on `family3_na025` at the identical protocol:
corridor `_trainlon` **0.939 / 0.939** (archived to three decimals; quoted as 0.93933 in the
E-043b harvest), corridor blended 0.937 / 0.938, `_holdlon` 0.933 / 0.933. The control is a
PAIR on purpose — a single monthly number would give the comparison no noise floor at all,
and the pair is the only reason the paragraph above can say what 0.001 means at monthly.

**WHAT THIS RUN CANNOT ANSWER, said before it runs.**

- **The gate is UNCERTIFIED BY CONSTRUCTION.** `GATE_REF_BY_CADENCE` has no pentad entry:
  `e017_u1_s0`'s 0.643 was measured by `ml/rollout.py` over the MONTHLY family-3 axis (#217),
  and it cannot certify a roll whose steps are a different length, whose starts are a
  different count and whose bands span different durations. The artefact will carry
  `"gate": {"pass": null, "skipped": true, "certified": false, "reason": …}` and every number
  from it is a **first reading**, in the §10 sense. The gate head is deliberately **not
  named** to this dispatch: it is a d_z-64 monthly head and `load_state_dict` against the
  d_z-32 pentad codec would kill the job.
- **Only `horizon_auc_daymatched` may be compared.** The raw `horizon_auc` averages 73 leads
  spanning 5–365 d against the archive's 12 spanning 30–365 d, most of them short, where
  skill is highest — a raw pentad number would beat the monthly one on lead SAMPLING alone
  (§7b(g)). Both are emitted; only the day-matched one is quotable against the control.
- **The intervals are not like-for-like.** At S = 3 there are **9** (start, year) samples
  behind each lead against monthly's **36**. The POINT ESTIMATES are comparable; the CIs are
  not, and no CI from this run may be set beside a monthly one below S = 12.
- **n = 1.** §3b's one-seed licence does not reach here — this changes codec, tensor and
  cadence at once. A seed-1 pentad head (E-044b-SEED1, below) must exist before any number
  here is written as a level.

#### 2 · The dispatch

```
"window": "recipe:xl144-zn-pentad-nolonhold,sroll:head-weights-e044b-xl144zn-pentad-s0,ckpt:run-415__pixelmae.pt,horizon:73,starts:3,dumproll",
"steps": "197428", "resume": "!run-415", "temporal_steps": "0",
"tensor": "family4_na025_pentad_r2", "anomaly": "true",
"runner": "gpu-box-46996216", "job_timeout": "1200"
```

**Three prerequisites, in order, and the roll cannot run before all three.**
(1) `publishtensor` on `gpu-box-39184683` → `family4_na025_pentad_r2_37e146384b.npz.{aa,ab,ac}`
on `data-cache-v1`, without which the roll box builds its own bytes, gets a different
fingerprint and **cannot find the published Z** (closed in the workflow at `ce361b0`, which
pulls and sha-verifies the pin). (2) A **HEADPUB** of #427's head off
`/opt/earth-cache/ckpt/temporal.pt` with `@temporal` — its 2.5 GB full checkpoint failed
**37** release uploads over the ~2 GiB asset cap, so it exists on one rented disk and in the
`probes-427` artifact only. (3) `ckpt:run-415__pixelmae.pt` — **not** `run-415.pt`, which is
not an asset name; a failed `curl` under `set -e` kills the script.

**Cost.** 3,363 roll steps = 441 scored + 2,922 long+future ≈ **14.1 h ESTIMATE at the
MONTHLY per-step rate**, plus ~10 min tensor pull, ~20–30 min X extraction (~34 GB, which is
why this is not on the 100 GB box) and ~2 min of dump upload ⇒ **≈15–16 h, ≈$4.6**. Every
hour of that is arithmetic over a rate measured on a different cadence: **nothing has ever
rolled the pentad tensor**, and the first `sroll` progress record settles it in minutes.
Read it before letting a 14-hour job run.

**Verification (§2), in order:** `axis: pentad · one step = 5 d · 12 months = 73 steps =
365 d` · `horizon: 73 steps (window token horizon:73)` · `starts: 3 per holdout year` ·
**`embed cache key: codec 8b639abe36 · tensor 37e146384b`** — the single most important line,
both halves must match or the box is holding the wrong tensor · `Z: … (3142, 86698, 32)
float16, 16.24 GiB` · `::warning::validation gate SKIPPED` · then `dump roll_…npz: 74 states
[86,698, 32] f16, 411 MB` per start, and at the end `gate: NOT CERTIFIED at pentad cadence`
and `dumproll: 9 trajectories, 3.70 GB`.

#### 3 · What rides along: the roll-forward sequences

Chris, 2026-08-22: *"Save the roll forward sequence for the held out years somewhere (so
that we can use it as animation in the UI)"*, and *"Roll forward all of the earth's pixels
(these are required by the stencil size, not just the relevant area)"*. The second half was
already true — `roll_step` advances all 86,698 window ocean pixels every step, because a
stencil head's step t+1 at a pixel reads its NEIGHBOURS at t, and gate/corridor/window are
masks applied to the decoded field afterwards. What was missing is that the state was
discarded one step after it existed.

`dumproll` writes one `.npz` per (head, holdout year, start): the full-window z trajectory
`[74, 86698, 32]` float16 (state 0 = the TRUE embedding of the start row), the axis
rows/labels/dates, the pixel index map, and the codec's `weight_hash` — **z, not pixels**,
because the decoder is published and deterministic and pixel space is ~20× the bytes at
C = 40. 9 files, ≈3.70 GB, in the probe artifact (30-day retention), NOT on `ml-metrics`.
The roll JSON is **byte-identical** with the flag on or off, which is what stops the
animation and the archived skill from being records of two different rolls.

---

<a id="e-044b-seed1"></a>
## E-044b-SEED1 · The replicate #427 owes — RESOLVED 2026-08-22: **both hypotheses hold.** #432 (E-044b-SEED1, pentad xl144+znoise stage-2 head, grad-clip 128, seed 1) ✓ GREEN 22:56Z on gpu-box-39184683, #427's own box

**RESULT (probes-432.json, harvested 23:30Z).** (a) Trained to 200,000 steps with **no
divergence** — with n=2 both seeds of the exact #423 configuration-plus-clip complete
where #423 died at 28k: **clipping-as-mechanism stands.** (b) One-step ratio
**0.50447** (10.729206 / 21.268318) vs #427's **0.50560** — the first measured pentad
pair spread, **|Δ| 0.00113**. Per §3b's 211M-row caveat this conflates training-seed with
val-draw and is an upper bound on the seed term; it is ~14× the monthly 211M pair's
0.00008 and still small in absolute terms, so the pentad one-step level **≈0.505 is now
pair-backed** — the pentad stage-2 task really is ~16× harder per step than monthly, and
E-044b-roll's collapse mechanism stands on a replicated footing. Integrity checks:
probe_head / raw-3×3 come back bit-identical to #427's (0.659 / 0.693) — CORRECT, these
score the frozen codec Z with fixed fold seeds (protocol determinism, not a replicate);
the pooled stage-2 probe spreads 0.583 / 0.630 (pair |Δ| 0.047, inside §3b's probe band,
labelled legacy). §3b's table gains the pentad forecast-ratio pair row in this commit.

Written at dispatch form, before the run exists (§1). Required, not optional: §3b's one-seed
licence needs all three of *scored by rolled corridor AUC*, *at the xl tier on the frozen f3
anchor and the monthly tensor*, and *effect ≥ 0.025*. #427 satisfies **none** of them — it
moves codec, tensor and cadence at once — so §3b's harder clause governs verbatim: *"Any new
metric, cadence, tensor, codec or scale tier with no measured pair. The first result at a
tier buys its own replication."* Code: `head_sha` **at dispatch**.

**E-044b-SEED1 · Trains a SECOND 206.5M xl144+znoise stage-2 head on the PENTAD tensor over
#415's frozen no-longitude-holdout pentad codec, IDENTICAL to #427 in every field but the
seed (`seed:0` → `seed:1`), with gradient clipping at 128 · params 206.5M head over a frozen
37.976M codec · stage `stage-2` · data `family4_na025_pentad_r2` (C 40, T 3,142, sha256
`37e146384b6f…`) · arch head 1024×16, K 24, stencil 145, ring `spiral:111,4444,0.71,0.5`,
znoise 0.7, grad-clip 128; codec 512×12, 4 heads, d_dec 256, d_z 32, patch 1 · steps×batch
stage-1 **0** (resumes at its own recorded step 197,428 — nothing trains) / stage-2
**200,000 × 256**, pool 251,337,502 windows · resume `run-415`.**

[E-044b · #427, the seed-0 arm this replicates](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-044b)

[E-044 · #423, which ran this configuration WITHOUT the clip and diverged](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-044-423-diverged)

#### 1 · Hypothesis and falsifier

**HYPOTHESIS, in two parts.** (a) With `--grad-clip 128` the seed-1 arm also trains to
200,000 steps **without divergence** — `stage2_grad_norm_max` bounded, `clip_frac` near 0,
`val_zmse` descending monotonically — which is what turns #427 from "one run that happened
not to blow up" into evidence that clipping was the mechanism. (b) Its one-step ratio
`z_t+1.mse_model / mse_persistence` lands **within a first-measured pair spread** of #427's
**0.50560** (10.746222 / 21.254400).

**FALSIFIER, two-sided and sharp.**
- **Divergence** — `stage2_grad_norm_max` climbing decade by decade, `clip_frac` rising off 0
  toward 1, `val_zmse` above 10× persistence at any window. Then **clipping was not the
  mechanism**, #423's other two candidates reopen in the order §0a set them: the **LR at
  pentad** first, the **under-scaled znoise** second (0.7 is 0.1512× this z-space's own
  one-step scale where E-036/E-037 measured 0.3979×). One arm cannot separate "clipping
  fixed it" from "two lucky draws", but two divergences out of two would settle it the
  other way.
- **A pair spread far above the monthly tier's.** The monthly 211M xl144 nolonhold pair
  measured **0.013918 (#414) / 0.013998 (#426), |Δ| 0.000080** on the identical quantity.
  A pentad |Δ| of that order is a reproducible level; one one or two ORDERS above it means
  the pentad one-step ratio is a draw, not a property, and every number quoted off #427 —
  including its corridor AUC — inherits that.

**CONTROL (§3).** #427 itself for (b), at 0.50560; the monthly pair above for the spread
that a "small" |Δ| would have to be read against. And the standing sentence that must travel
with every quotation of either: **the pentad ratio is NOT comparable to the monthly arm's
0.0139 as a level** — a 5-day step in a patch-1 z-space is a far harder one-step problem
than a monthly step in a patch-3 one, and the two differ in cadence, codec, patch size and
`d_z` at once. What is comparable is pentad against ITSELF across seeds, which is exactly
what this arm buys.

#### 2 · The dispatch, and why it costs no embed

Identical to #427's validated 25-field block (spec §0c) with **one** field changed —
`seed:0` → `seed:1` in the window's tail — plus the `runner`. `--grad-clip 128` stays; the
`sched:` tail goes LAST and `scripts/probes_run.sh` word-splits it into the `python -u
ml/temporal.py` line.

**NO EMBED PASS, and this now holds on a box that has never seen the tensor** — which was
not true before tonight. Two mechanisms, both verified in the code rather than assumed:

1. **The stage-2 path DOES pull a published embed cache before deciding to embed.**
   `scripts/probes_run.sh:379` runs `python -u ml/embed_cache_sync.py pull --run actions
   --data "$TENSOR"` — guarded `|| echo`, because a miss is the normal path — and it sits
   **before** the trainer at `:398`. `ml/temporal.py:1379` names the cache
   `embed_cache_path(run, whash, dhash)`, and `embed_everything` short-circuits at
   `:858-862`: an existing file of the right `(T, P, d_z)` prints `(cached: …)` and returns.
   #427's own log is the proof — `embed cache already local and valid: (3142, 86698, 32)
   float16, 16.24 GiB`, zero `{"embedding": …}` records.
2. **The cache key is now reproducible off-box.** The key is (codec weight hash, tensor
   sha256) = `8b639abe36` / `37e146384b`. The codec half travels through the release
   (`run-415__pixelmae.pt`, seeded by the workflow's own step; verified locally —
   `codec_weight_hash` of that asset **is** `8b639abe36`). The tensor half could NOT travel
   until `ce361b0`: `np.savez` stamps zip timestamps, so a rebuilt tensor never hashes the
   same, and the Build step now **pulls the pinned tensor** and sha-verifies it. #427 pushed
   the Z itself (`embed cache for codec 8b639abe36/37e146384b is now durable`, 12 chunks,
   17,433,927,552 B on `embed-cache-v1`).

So this arm's cost is **train only: ≈20.6 h at #427's 0.371 s/step, ≈$6.5**, against the
≈24–26 h and ≈$7.5–8 #423 paid with its embed. It also inherits **§0f's 24-hour
`GITHUB_TOKEN` ceiling**: at ~21 h it is close enough that the harvest should be planned off
`ml-live-<n>` and the artifacts by hand.

**BOX — UNRESOLVED AT DRAFTING, and it is the one blocker.** The arm needs a **126 GB RAM**
machine: `temporal.py` keeps the tensor eager, `np.nan_to_num(X)` copies 34 GB beside the
34 GB `X` and `np.isfinite(X)` adds a 17 GB bool — an ~85 GB peak before `del X`, and it
dies as exit 137 with no message. At drafting time **the repo has four registered runners
(`gpu-box-30257785`, `31479844`, `39184683`, `46996216`) and ALL FOUR ARE OFFLINE**;
`gpu-box-47094145` is **not registered at all**. A dispatch naming a runner that does not
exist queues forever. Start (or create) the box and confirm it appears in
`node scripts/fleet_run_state.mjs` BEFORE dispatching, and check the disk: this arm needs the
4.5 GB tensor + the 17.4 GB Z + ~50 GB of resident tensor in RAM, but **not** the 34 GB `X`
extraction the roll needs.

**First minutes (§2), unchanged from #427's list except item 0:** `runner_name` is the box
you named · the recipe block resolves with `RECIPE_HOLDOUT_LON=0,0` · **stage 1 trains
nothing** (`config.steps` 197428 = `resumed.at_step` 197428 — anything else means the codec
is being modified AND the Z cache key with it: cancel within seconds) · **`(cached: …)` and
ZERO `{"embedding": …}` records** — a restart here is ~8.5 h and ~$2.6 and means the tensor
pull or the codec seed did not produce the pinned pair · `params_M` **206.536** ·
`train_windows` **251,337,502** · `val_persistence` **21.44622**, identical to #427/#423 to
five decimals, which is the tightest possible check that this is the same Z · **`gradient
clipping ON: max_norm 128`** — if it says OFF the `sched:` tail lost the flag, and this is
#423 again: cancel · `stage2_config.seed` **1**, which is the ONLY field that distinguishes
this run's artefacts from #427's.

---

<a id="fleet-2026-08-21-0740"></a>
## OPERATIONS · The fleet at 2026-08-21 07:40Z — two boxes, two jobs, one box stopped, burn down 25%

Not an experiment. Recorded because it sets what the next check-in inherits (§0e: the
arithmetic, early — nothing here was scaled down to fit).

| box | $/h | job | state at 07:40Z |
|---|---|---|---|
| `gpu-box-31479844` (vast 47724559, Quebec) | 0.294 | **#426** (E-043b-SEED1) at this hour; **#428 / #429 now** | Reading at 07:40Z: stage 2 step **80,000 / 200,000**, `val_zmse` **0.07085** / persistence 3.09512 = ratio **0.02289**, grad norm **0.3003**, amp 0.9913 — monotone, healthy, worst norm all run 2.99. 0.2776 s/step ⇒ ends ~16:50Z. **SINCE LANDED: #426 (E-043b-SEED1) completed 16:59:11Z, 15 h 46 min, ≈$4.64.** The box was stopped at 16:59Z, restarted 17:46:04Z, and its current occupants are **[#428 (HEADPUB e043b-xl144-nolonhold-s1) and #429 (E-043b-SEED1-roll)](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043b-seed1-roll)** |
| `gpu-box-39184683` (vast 47724565, Hong Kong) | 0.308 | **#427** (E-044b) | stage 2 step **2,000 / 200,000** at 0.258 s/step ⇒ ends **~00:30Z on the 22nd**. **The 8.5 h embed was REUSED — zero embedding records.** `grad_clip` 128.0, `clip_frac` 0.0245, `grad_norm_max` **452.0** |
| `gpu-box-46996216` (vast 47913006, Austria) | — | — | **STOPPED at 05:53Z**, not destroyed. #419 finished successfully at 05:22:58Z and had been burning idle for 30 min. Its 213 GB of daily tensor is intact on the disk |

**Money.** Credit **$35.56**, burn **$0.9374/h** ⇒ runway **37.9 h**, i.e. to **~21:30Z on
2026-08-22**. The burn decomposes as **$0.6022/h of compute on the two running boxes** and
**$0.3352/h of storage on ten STOPPED ones** — stopping #419's box cut the total from
$1.2574/h, a **25% reduction**, and **36% of what remains still buys no computation at all**.
The stopped-box sweep is now the largest single lever on this programme's burn and it has
been deferred twice.

| what is outstanding | needs | finishes |
|---|---|---|
| **#427** (E-044b) | ~18–19 h, ~**$5.7** | ~00:30Z on 2026-08-22 |
| ~~**#426** (E-043b-SEED1)~~ **DONE** | estimated ~9.3 h, ~**$2.7** from here; the whole job was **15 h 46 min, ≈$4.64** | **completed 16:59:11Z on 2026-08-21** |
| **#426**'s headpub + roll (§5 of that entry) — dispatched as **[#428 + #429 (E-043b-SEED1-roll)](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043b-seed1-roll)** | ~3.7 h, ~**$1.1** — #428 done in ~6 min ≈$0.03, #429 ~3.5 h ≈$1.03 | **#429 in flight, ≈21:40Z on 2026-08-21** |
| the pentad `sroll:` over #427's head | ~14 h, ~$4.3 | ~15:00Z on 2026-08-22 |

**Everything fits inside the runway**, the pentad sroll with ~6 h to spare.

**TWO SINGLE POINTS OF FAILURE, both on `gpu-box-39184683` (vast 47724565), both fixable by
publishing.** (1) The **16.24 GiB pentad Z** — 8.5 h of GPU — exists on that one rented disk
and nowhere else; it was never pushed to `embed-cache-v1`. (2)
**`family4_na025_pentad_r2.npz`** (4.5 GB) is published nowhere and that box is the only one
that has ever built it, so any other box reinstates the **E-008 box effect** (0.041 on the
head k-fold at a fixed seed). **DO NOT DESTROY 47724565.** Publishing both is the work that
would make the seed-1 arm and the §7a sroll dispatchable anywhere.

**A THIRD, newly found and structural: a job over 24 hours cannot archive its own results.**
See the #419 note in `claude/E044-pentad-stage2-spec.md` §0f. **#419 ran 35.96 h**, its
`Archive metrics` and `Upload probe results` steps both failed on an expired `GITHUB_TOKEN`
(`401 Bad credentials`, `Authentication failed`) and **both reported success** — §4.6
exactly. Both fallbacks held (the live branch was kept rather than deleted, the artifact
uploaded), and `run-419.jsonl` and `probes-419.json` were **rescued by hand into
`ml-metrics`** at ~06:00Z on the 21st. The durable fix is a workflow change: those steps need
a PAT rather than the job token on any job that can exceed 24 h. Until then, **harvest every
long run by hand and never read a green archive step as evidence** that anything was
archived.

**#419's own result, recorded here because its entry cannot rely on its artifacts.**
E-043f, the fresh 38.0M **daily** codec (`f5-40M-nolonhold`, `family5_na025_daily`, T
15,706), completed **successfully** — Train **33.11 h**, Probes **2.70 h**, total 35.96 h. It
was slow, never wedged. **`probe_head.json` is ABSENT from its bundle** — the fourth run in
this wave to lose its head number — so the daily codec's **unpooled** verdict does not exist
and cannot be quoted. Its pooled k-fold, written as the pooled number §3 says to distrust at
this cadence: rapid `r_kfold_deseas` **0.612** [0.563, 0.659] n 7,290, `rmse_sv` 3.26,
against a **wind-only baseline of 0.607** [0.547, 0.664] on the same folds — i.e. at daily
cadence the codec and the wind-stress ridge are indistinguishable on this read-out. `fc`
0.364 [0.287, 0.432] n 13,613 against a wind-only 0.110, where it is clearly ahead.
`dip_check` r_out_of_fold 0.579, dip captured **25.6%** (against #415's pentad 31.8%).
**The full entry now exists**, with the ladder, the fired falsifier, the cost split and this
archival failure written up:

[E-043f · #419 (fresh 38.0M daily codec, all longitude columns) — the RESULT](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043f)

---

<a id="e-044b"></a>
## E-044b · #427 — DISPATCHED 2026-08-21 05:54:12Z. The same arm as #423 with gradient clipping, and the first log window already shows what #423 could not

Written **at dispatch**, hypothesis and falsifier first (§1); the first-minutes verification
below was appended as each item landed, not rewritten afterwards. Governed by
`claude/E044-pentad-stage2-spec.md` (§0a–§0f are new and carry the diagnosis).
Code: `head_sha` **aced980**.

**E-044b · Trains a 206.5M xl144+znoise stage-2 head on the PENTAD tensor over #415's frozen
no-longitude-holdout pentad codec, with the stage-2 training pool open to every longitude and
years-only holdout, WITH GRADIENT CLIPPING AT 128 — the one field changed from #423, which
ran this exact configuration and diverged (re-dispatch of #423) · params 206.5M head over a
frozen 37.976M codec · stage `stage-2` · data `family4_na025_pentad_r2` (C 40, T 3,142,
sha256 `37e146384b6f…`) · arch head 1024×16, K 24, stencil 145, ring
`spiral:111,4444,0.71,0.5`, znoise 0.7, **grad-clip 128**; codec 512×12, 4 heads, d_dec 256,
d_z 32, patch 1 · steps×batch stage-1 **0** (resumes at its own recorded step 197,428 —
nothing trains) / stage-2 **200,000 × 256** · resume `run-415`.**

[#427 (E-044b pentad xl144+znoise stage-2 head, with grad clipping) — the live status page](https://blauewelt.github.io/earth/status.html#run-427)

[#427 — the CI log](https://github.com/blauewelt/earth/actions/runs/32452280284)

[E-044 · #423, the run this replaces, and the full diagnosis](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-044-423-diverged)

[E-044 · the dispatch spec (project doc)](https://blauewelt.github.io/earth/docs.html?f=claude/E044-pentad-stage2-spec.md)

#### 1 · Hypothesis and falsifier

**ONE FIELD CHANGED.** `--grad-clip 128` appended to the window's `sched:` tail. LR (1e-3),
warmup (2,000), schedule (expdecay, halflife 40k), seed (0), znoise (0.7), stencil, ring and
every architecture field are identical to #423, so this is a **one-variable test against
#423's own trace** rather than a new experiment.

**HYPOTHESIS.** #423's divergence was the missing clip. With it the arm trains to 200,000
steps with the pre-clip grad norm in the 8-ish band and `val_zmse`/persistence descending
past #423's best of 0.5404.

**FALSIFIER, and it is sharp precisely because only one field moved.** If the run still
leaves the healthy band — `stage2_grad_norm_max` climbing decade by decade,
`stage2_grad_clip_frac` rising off 0 toward 1, `stage2_val_zmse` never beating 11.58984 —
then clipping was **not** the mechanism, and the remaining candidates are the LR at pentad
and the under-scaled znoise, **in that order**. A **separate** falsifier applies to the
THRESHOLD rather than the mechanism: a `clip_frac` that saturates at 1.0 would mean 128 is
setting the effective learning rate instead of the schedule, and the number is wrong even if
the idea is right.

**WHAT THIS RUN CANNOT ANSWER, said at dispatch.** It cannot separate "clipping fixed it"
from "the run got a luckier draw", at n = 1 — what it can do is show the mechanism operating,
which the instrumentation below now does directly. And it says nothing about whether znoise
0.7 is the right level at pentad: §0a(2) of the spec shows it is **0.1512 ×** this z-space's
own one-step scale where E-036/E-037 measured **0.3979 ×**, so this arm carries that
finding's NUMBER and not its INTERVENTION. "znoise at pentad" remains an OPEN axis with no
measurement at all.

#### 2 · First-minutes verification — MEASURED, in order (§2)

| # | item | reading | |
|---|---|---|---|
| 1 | `runner_name` | `gpu-box-39184683` | **PASS** |
| 2 | recipe resolved | `recipe: xl144-zn-pentad-nolonhold` in the config record | **PASS** |
| 3 | **stage 1 trains nothing** | `config.steps` **197428** = `resumed.at_step` **197428**, `from` `run-415.pt` | **PASS** |
| 4 | Train step wall | **0.23 h** (13.8 min), no re-fit | **PASS** |
| **0** | **the embed did NOT restart** | **ZERO `{"embedding": …}` records on `ml-live-427`.** `stage2_config` appeared 67 min after the `Probes` step began, which is the tensor load + anomaly transform + probe ladder and no embed at all | **PASS — 8.5 h and ~$2.6 saved** |
| 8 | the Z is the SAME Z | `val_persistence` **21.44622** — identical to #423 to five decimals, on the same 4,096 monitor windows | **PASS** |
| 10 | head size | `params_M` **206.536** (= 206,535,712), `d_z` **32** | **PASS** |
| 6 | window pool | `train_windows` **251,337,502** | **PASS** |
| 5 | holdout | `codec_holdout_lon` `0,0`, `train_lon_hold` `none` | **PASS** |
| **13** | **clipping is ON** | `stage2_config.grad_clip` **128.0** | **PASS** |
| **15** | znoise is recorded at last | `input_znoise` **0.7**, `input_znoise_rel_pers` **0.15116**, `input_znoise_rel_zrms` **0.13518**, `z_rms` **5.17825** | **PASS** |
| 11 | LR non-zero | `stage2_lr` **9.9998e-4** at step 2,000 | **PASS** |
| 12 | step rate | **0.258 s/step** (516.9 s / 2,000) — better than the 0.27 ESTIMATE and than #423's probe-inclusive 0.371 | **PASS** |

**The `--grad-clip` flag reaches the trainer through the same route `--input-znoise` does:**
`sched:` goes last, and `scripts/probes_run.sh` builds
`SCHED="--lr-schedule ${WINDOW##*sched:}"` and word-splits it unquoted into the
`python -u ml/temporal.py` line. Verified by resolving the recipe locally before dispatch,
and confirmed after it by `grad_clip: 128.0` appearing in `stage2_config`.

#### 3 · THE FIRST LOG WINDOW ALREADY SETTLES THE DIAGNOSIS

```
{"stage2_step": 2000, "stage2_zmse": 6.40817, "stage2_val_zmse": 11.31759,
 "stage2_amp": 0.8235, "stage2_grad_norm": 8.7887, "stage2_lr": 0.0009999826714706267,
 "stage2_wall_s": 516.9, "stage2_grad_clip": 128.0, "stage2_grad_norm_max": 452.0087,
 "stage2_grad_clip_frac": 0.0245, "stage2_grad_nonfinite": 0}
```

**Read `grad_norm_max` against `grad_norm`.** The sampled pre-clip norm is **8.7887** — the
number #423 also published at this step (8.2372), and the number every check-in read as
healthy. The **worst step in the same 2,000-step window was 452.0087**, and **2.45% of the
window's steps — about 49 of them — exceeded 128 and were clipped.**

**So #423 was never in a healthy regime.** It was drawing gradients fifty times its own
sampled norm from step one, at a rate of one step in forty, and **the one-step-in-2,000
sampling was structurally blind to every one of them**. This is §4.10's question — *what
would look identical whether this works or fails?* — answered on the first record the new
instrumentation ever produced. The quantity that distinguishes the stories is not the grad
norm; it is the grad norm's **tail**, and nothing was measuring it.

It also confirms the mechanism claimed in the diagnosis rather than merely being consistent
with it: a window pool of 251M pentad windows produces rare, enormous gradients at a
measurable rate, and the only thing standing between that rate and an AdamW second moment is
a clip.

**And the threshold is in the right regime.** `clip_frac` 0.0245 is neither 0 (which would
mean 128 never binds and buys nothing) nor near 1 (which would mean the clip, not the
schedule, is setting the learning rate — the separate falsifier in §1). It bites the tail
and leaves the bulk alone, which is what it was sized to do.

**Early comparison against #423 at matched steps**, first reading, one seed:

| step | #423 val/pers | **#427 val/pers** | #423 grad norm | #427 grad norm | #427 window max | #427 clipped |
|---|---|---|---|---|---|---|
| 2,000 | 0.5404 | **0.5277** | 8.2372 | 8.7887 | **452.01** | **2.45%** |
| 4,000 | 0.5509 | **0.4969** | 8.2483 | 5.5304 | **11.98** | **0.0%** |
| 6,000 | **1.0453** | **0.5429** | **787.21** | **9.0505** | **11.64** | **0.0%** |

**At step 4,000 #427 is already better than #423 ever got** — 0.4969 against #423's
lifetime best of 0.5404 — and the second window is a textbook "healthy, never binds"
reading: **maximum 11.98, nothing clipped**, against the first window's 452.01 and 2.45%.

**Read the two windows together, because that contrast is the finding.** The tail is
real and it is concentrated in the LR warmup (which reaches 1e-3 at step 2,000); once past
it the pre-clip norm never approaches the threshold. #423 ran the same seed on the same
data and therefore met the same 452-class steps — **unclipped** — and its published norms at
2,000 and 4,000 (8.24, 8.25) said nothing about them at all. **Step 6,000 remains the
decisive test**: that is where #423 left the band and never returned.

#### 3b · STEP 6,000 — THE DIVERGENCE DID NOT HAPPEN

```
{"stage2_step": 6000, "stage2_zmse": 4.24833, "stage2_val_zmse": 11.64337,
 "stage2_amp": 0.866, "stage2_grad_norm": 9.0505, "stage2_wall_s": 1501.4,
 "stage2_grad_clip": 128.0, "stage2_grad_norm_max": 11.6429,
 "stage2_grad_clip_frac": 0.0, "stage2_grad_nonfinite": 0}
```

Side by side at the step where #423 broke — **same seed, same data, same schedule, same
z-space, one field different**:

| | #423 (no clip) | **#427 (`--grad-clip 128`)** |
|---|---|---|
| `stage2_zmse` | 17.70444 | **4.24833** |
| `stage2_val_zmse` | 22.41752 | **11.64337** |
| val / persistence | **1.0453** | **0.5429** |
| `stage2_grad_norm` | **787.21** | **9.0505** |
| worst step in the window | *not measured* | **11.6429** |
| steps clipped in the window | *no clip existed* | **0.0%** |

**#427 passed the step at which #423 left its band and never returned, with a grad norm of
9.05 against 787.21 and a whole window whose worst step was 11.64.** The hypothesis of §1 is
supported and no falsifier has fired: `grad_norm_max` is not climbing, `clip_frac` is not
rising off zero, and `val_zmse` has already beaten 11.58984.

**What this is and is not.** It is **one seed**, 6,000 steps of 200,000, and §3b's harder
clause applies verbatim — a seed-1 arm is required before any number here becomes a level.
It is also not, by itself, proof that clipping is the *only* thing that was wrong: what it
shows is that the mechanism operated (the tail is real, 2.45% of the warmup window needed
the clip and got it) and that the failure it was aimed at did not recur under an otherwise
identical configuration. The remaining 97% of the run is what turns that into a result.

#### 4 · The z-space, measured for the first time

`z_rms` **5.17825** is the first absolute measurement of a stage-2 z-space in this
programme's history — no archived run carries the field, which is why #423's diagnosis had to
derive the pentad scale indirectly from the codecs' stage-1 `z_mse_persistence` records.
Beside it, `sqrt(val_persistence)` = **4.6310**.

**The ratio is 0.894, and it is worth staring at.** For a stationary series
E[(z_t − z_{t+1})²] = 2σ²(1 − ρ), so a one-step change 0.894 × the size of the field itself
implies a **five-day lag-1 autocorrelation of ρ ≈ 0.60**. The pentad z-space substantially
decorrelates in five days. That, and not any defect of the head, is why the pentad headline
ratio sits near 0.5 where the monthly arm reaches 0.032 — **the two numbers are answers to
different questions and must never be quoted as a ladder.** The monthly counterpart of
`z_rms` does not exist anywhere in the archive; the field now exists, so the next monthly
stage-2 run will produce it and the comparison can finally be made properly.

`input_znoise_rel_pers` **0.15116** confirms, from the run's own data, the **0.1512** that
was derived indirectly before dispatch.

#### 5 · Budget and the thing that will bite at the end

| | |
|---|---|
| stage-2, 200,000 × 0.258 s/step | **14.3 h** |
| in-training transport probes (10 × 3,119 forwards) + eval-3 | ~2.5 h |
| already spent (tensor load, anomaly transform, probe ladder, stage-1 no-op) | 1.5 h |
| **total job** | **≈ 18–19 h from 05:54Z ⇒ ends ~00:30Z on 2026-08-22** |
| cost at $0.308/h | **≈ $5.7** |

`job_timeout: 2400` (40 h) has ample margin. **§0f of the spec does not**: a GitHub Actions
`GITHUB_TOKEN` expires at 24 hours, #419 lost both its archive steps to exactly that while
they reported success, and #427 at ~19 h is inside the ceiling but not comfortably.
**Harvest `ml-live-427` and its artifacts by hand regardless, and do not read a green
`Archive metrics` step as evidence that anything was archived (§0.2).**

#### In-flight reading, 2026-08-21 17:35Z — the clip fix WORKED and the run is nevertheless not learning

**The fix is confirmed.** `--grad-clip 128` caught the startup spike exactly where **#423
(the E-044 pentad stage-2 first attempt, CANCELLED on divergence)** blew up:
`stage2_grad_clip_frac` **0.0245** in window 1 (step 2,000) with `stage2_grad_norm_max`
**452.0087**, then **0.0 for all 67 subsequent windows**; `stage2_grad_nonfinite` **0
everywhere**; `norm_max` max over windows 2–68 is **98.7448** (step 10,000), decaying to a
stable **~3.7–4.0** band from ~step 24,000 and reading **3.7136** last. **No numerical
instability at any point.**

[E-044 · #423, the divergence this arm was built to fix](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-044-423-diverged)

**And the run has saturated.** `val_persistence` **21.44622**. Normalised
`stage2_val_zmse`: **0.4893** at step 36,000 → **0.4853** at 70,000 → **0.4867** at 102,000 →
**0.4901** at 136,000. **Best-ever 0.4828 at step 66,000**, and the latest reading is above
it and above the 0.49 health bar. Of 67 transitions, **34 are increases** — a
noise-dominated flat curve, not a descending one. Corroborating and worse: `rapid_r_deseas`
falls at **five of its six** probes — 0.596 (20k), 0.576, 0.584, 0.559, 0.543, **0.526
(120k)** — a drift of **−0.070 across the run** with a single uptick at the third reading.
(An earlier draft of this line said "monotonically"; it is not, and the numbers beside it
say so. §0.1 — the claim is corrected here rather than the series being trimmed to fit it.)

**The decision, and its reason. The run is NOT cancelled.** `scripts/probes_run.sh:428`
pushes the **17.43 GB** pentad embedding cache `Z` to the `embed-cache-v1` release
unconditionally but **only AFTER all 200,000 steps** — and that cache, **8.9 h of a 4090**,
exists in exactly one place: box 47724565's local disk. Commit `93f1fc2` fixed the
container-wide `/tmp/embed-cache-pushed` marker bug that had killed the in-training push, but
**that fix takes effect on the NEXT job, not on #427**. Cancelling would forfeit the only
backup route for the single largest single-copy artefact in the programme. The remaining
~4.3 h and ~$1.3 buy that backup, plus a full-budget curve for a cadence §3b records as
entirely **UNMEASURED**.

**What it means for the next arm.** The E-044 question is **no longer clipping — that is
settled.** It is **regularisation and step budget**: a 206.5M head at `input_znoise` 0.7 on
`family4_na025_pentad_r2` reaches its best at roughly a third of its budget and then drifts.
**Open follow-up:** the milestone heads should be checked, and the **step-60,000 milestone,
not the final head, may be the one worth rolling.** Stated plainly: whether #427's milestone
steps include 60,000 was **NOT verified in this session**.

**The remaining arithmetic.** Measured rate **243.97 ms/step** over steps 126,000 → 136,000;
64,000 steps remaining ⇒ **4 h 20 min**, plus ~**1,682 s** of probe overhead at 140k / 160k /
180k ⇒ stage-2 end **≈ 22:23–22:33Z**, job end **≈ 22:40Z ± 20 min**.

---

<a id="e-044-423-diverged"></a>
## E-044 · #423 DIVERGED AND WAS CANCELLED — stage 2 had no gradient clipping, and nothing was watching the norm it published

**#423 (E-044 pentad xl144+znoise stage-2 head) · what it did, in absolute terms: trained a
206.5M xl144+znoise stage-2 head on the PENTAD tensor over #415's frozen
no-longitude-holdout pentad codec and diverged from its step-2,000 best · params 206.5M head
over a frozen 37.976M codec · stage `stage-2` · data `family4_na025_pentad_r2` (C 40,
T 3,142) · arch head 1024×16, K 24, stencil 145, ring `spiral:111,4444,0.71,0.5`,
znoise 0.7, NO gradient clipping; codec 512×12, d_z 32, patch 1 · steps×batch stage-1 0 /
stage-2 **28,000 of 200,000 × 256, CANCELLED 2026-08-21 04:39:42Z** · resume `run-415`.**

Cancelled by the session at ~04:39Z. `head_sha` 8f0e514 (the code #415 and #423 both ran).

### 1 · The full stage-2 trace, from `ml-live-423`

`stage2_monitor.val_persistence` **21.44622**; `stage2_config.train_windows` **251,337,502**;
`params_M` **206.536**; `d_z` **32**; `codec_holdout_lon` `0,0`; `train_lon_hold` `none` —
every pre-registered first-minutes item PASSED. The run was the run it said it was. It still
failed.

| step | zmse | val_zmse | val/persistence | amp | grad norm | lr |
|---|---|---|---|---|---|---|
| 2,000 | 6.77614 | 11.58984 | **0.5404 ← BEST** | 0.8218 | **8.2372** | 1.000e-3 |
| 4,000 | 5.26661 | 11.81387 | 0.5509 | 0.8372 | **8.2483** | 9.659e-4 |
| 6,000 | 17.70444 | 22.41752 | 1.0453 | 0.8568 | **787.21** | 9.330e-4 |
| 8,000 | 16.90655 | 19.25227 | 0.8977 | 0.8751 | 3,891.03 | 9.012e-4 |
| 10,000 | 17.99294 | 20.56906 | 0.9591 | 0.8708 | 2,928.00 | 8.705e-4 |
| 12,000 | 121.66760 | 220.34981 | **10.274** | **2.8252** | 6,277.70 | 8.409e-4 |
| 14,000 | 20.92885 | 24.34108 | 1.1350 | 0.9110 | 2,078.22 | 8.122e-4 |
| 16,000 | 18.11284 | 21.52560 | 1.0037 | 0.9670 | 2,998.00 | 7.846e-4 |
| 18,000 | 13.54183 | 15.51535 | 0.7235 | 0.7660 | 1,106.62 | 7.578e-4 |
| 20,000 | 13.53046 | 15.21943 | 0.7097 | 0.7338 | 1,026.88 | 7.320e-4 |
| 22,000 | 13.39618 | 15.03861 | 0.7013 | 0.7529 | 1,168.29 | 7.071e-4 |
| 24,000 | 120.73129 | 226.75468 | **10.573** | **2.8541** | **13,051.75** | 6.830e-4 |
| 26,000 | 18.93076 | 20.89788 | 0.9744 | 0.7805 | 6,958.42 | 6.597e-4 |
| 28,000 | 19.33818 | 20.20434 | 0.9421 | 0.9508 | 6,973.99 | 6.372e-4 |

In-training probe at 20,000: `rapid_r_deseas` **0.579** (section-POOLED — emitted for
protocol determinism, excluded from every claim, §7 of the spec).

**Read the shape, not the levels.** The best validation is at step **2,000** and nothing
after it comes close. After step 6,000 the grad norm **never returns below 1,000** — this is
a sustained regime change, not a sequence of isolated spikes. And the two blow-ups are
nearly the SAME state (zmse 121.67 / 120.73, val 220.35 / 226.75, amp 2.825 / 2.854), which
says the run was oscillating in and out of one degenerate configuration rather than being
hit by two unrelated bad batches.

**The healthy comparator, same code, same architecture, monthly tensor.** #426
(E-043b-SEED1) at step 52,000: `val_zmse` **0.09919** / persistence **3.09512** = ratio
**0.03205**, grad norm **0.7274**, amp 0.9877, monotone. Its worst norm all run is 2.99.

### 2 · Diagnosis — three candidates, and only one of them is load-bearing

Established from artefacts, not from the shape of the failure.

**(a) THE Z-SCALE DIFFERENCE IS REAL, REPLICATED — AND IS NOT WHAT BROKE THE RUN.**

First, the comparison that was in doubt is sound as it stands. `val_persistence` is
`(Z[t] - Z[t+1]).pow(2).mean()` — `ml/temporal.py:1814`, a **mean, not a sum** — so it is
already a PER-COMPONENT quantity and **`d_z` 32 vs 64 and C 40 vs 39 do not enter it**.
21.44622 against 3.09512 is apples to apples: the pentad z-space's RMS one-step change is
**sqrt(21.44622) = 4.631** against the monthly anchor's **sqrt(3.09512) = 1.759**, a factor
**2.63**.

That factor is confirmed independently, off the CODECS' own stage-1 probe records on
`ml-metrics`, in two pairs:

| codec | tensor | d_z · patch | final `z_mse_persistence` | RMS |
|---|---|---|---|---|
| **#62** (f3 anchor, monthly) | `family3_na025` | 64 · 3 | 2.953 | 1.719 |
| **#63** (f3 anchor, seed 1) | `family3_na025` | 64 · 3 | 3.298 | 1.816 |
| **#386** (E-038a, pentad r1) | `family4_na025_pentad` | 32 · 1 | 20.98 | 4.581 |
| **#415** (E-043e, pentad r2 — this codec) | `family4_na025_pentad_r2` | 32 · 1 | **19.875** | **4.458** |

Two monthly codecs and two pentad codecs, agreeing within their pairs and separated by
2.6× across them. It is a property of the pentad z-space, not of #415. (It is also not
inherited: at step 0, before training, the same figures read 0.167 / 0.154 monthly against
0.626 / 0.841 pentad — patch-1 encoding does no 3×3 spatial averaging, so its z is
heavier-tailed from the start, and training then expands both.)

**And yet the scale cannot be the destabiliser, because AdamW does not see it.** Multiply
every gradient by a constant and AdamW's update `m̂/(√v̂ + ε)` is unchanged; the decoupled
weight decay is unchanged too. Measured rather than asserted — 400 AdamW steps, same seed,
loss multiplied by **6.929** (the exact `val_persistence` ratio):

```
A · PURE LOSS RESCALE  L -> 6.929 L   (gradients 6.929x, same problem)
   grad-norm ratio step 0 / step 399 : 6.9290 / 6.9290
   max |dparam| after 400 AdamW steps: 3.973e-05   (param scale 0.3054)  -> relative 1.30e-04
B · TARGET RESCALE     y -> 2.632 y   (a genuinely 2.632x larger z-space)
   grad-norm ratio step 0 / step 399 : 2.5848 / 3.7264
   max |dparam| after 400 AdamW steps: 1.036e-01   -> relative 3.39e-01
```

So: **`stage2_grad_norm` reading 8.24 at pentad where the monthly arm reads 1.24 is a
statement about the units of the z-space, not evidence of pathology** — 8.2372 and 8.2483
at steps 2,000 and 4,000 are as stable a pair as the archive contains. Row B says the
z-scale is not *inert* either (a larger target is a genuinely different optimisation
problem, 34% different trajectory in the same toy) — but nothing in it predicts an episodic
1,600× excursion, and **a lower learning rate is therefore NOT indicated**: lr 1e-3 is
healthy at monthly, and it was healthy at pentad for 4,000 steps.

**(b) `--input-znoise 0.7` IS MIS-SCALED — AND IN THE BENIGN DIRECTION, SO IT IS NOT THE
CAUSE EITHER.** The flag adds `randn_like(z4) * 0.7` to live slots
(`ml/temporal.py:1936`): an **absolute** sigma, in whatever units the frozen codec happens
to emit. E-036/E-037 sized 0.7 on the MONTHLY z-space, from that arm's own one-step error
(`sqrt(val_zmse) ≈ 0.74`), and nothing has ever checked that it transfers.

| | monthly anchor (where 0.7 was measured) | #415's pentad codec (where #423 used it) |
|---|---|---|
| sqrt(`val_persistence`) | 1.7593 | 4.6310 |
| `--input-znoise 0.7` as a fraction of it | **0.3979** | **0.1512** |

**The same number is a 2.63× WEAKER perturbation at pentad.** Two consequences, and they
point in different directions. It is *not* the divergence — weaker input noise does not
explode gradients, and if anything it removes smoothing. But it is *also not the
intervention E-036/E-037 measured* (+0.045/+0.050 corridor AUC, the largest replicated
stage-2 effect in the log): #423 and its successor carry the monthly **number**, not the
monthly **perturbation**, and no claim off this arm may say otherwise. §3b forbids carrying
a level across a tier boundary and this is exactly that, caught after the fact.

**(c) THE LOAD-BEARING FACT: there was no gradient clipping, and one unclipped step is
worth a thousand.** Until 2026-08-21,
`grep -n "clip_grad\|max_norm\|grad_clip" ml/temporal.py` returned **nothing**. AdamW
bounds the update per COORDINATE, not the damage per STEP: one outlier batch spikes m and v
together, and the second moment then stays inflated for ~1/(1−β₂) = 1,000 steps. In the
numbers this run produced, with 8.25 as its own healthy norm:

- **unclipped at its worst, 13,051.75 = 1,582× healthy** → v grows **2,503×**, √v **50×**, so
  every honest gradient for the next ~1,000 steps is divided by 50;
- **the same step clipped at 128.0 = 15.5× healthy** → v grows **1.24×**, √v **1.11×**.

That is the whole difference between a run that recovers and one that does not, and it is
exactly the shape #423 shows: a spike, thousands of steps of partial recovery, another
spike, and a validation curve that never returns to step 2,000.

**Verdict. Load-bearing: the missing clip.** Red herrings, both quantified above rather than
waved away: the z-scale (real, replicated, invisible to AdamW) and `d_z`/channel-count
(they do not enter a per-component mean at all). Mis-scaled but not causal, and now
recorded rather than assumed: the znoise. Structural and unchanged, listed so the next
reader does not re-derive them: K = 24 pentads is **120 days** of context against 24 months'
730, the window pool is **251,337,502** against 38,488,680 (6.53×, so 200k × 256 = 51.2M
draws covers 20% of the pentad pool once against 1.33 passes of the monthly one), and
`d_z` is 32 against 64. None of these is a stability mechanism; the first two are why the
pentad headline ratio (best 0.540) is not comparable to the monthly one (0.032) and must
never be quoted against it without that sentence attached.

### 3 · The cost, and what the cancel preserved

| | |
|---|---|
| embed pass (8.5 h, 30,573 s to 95.6% + the tail) | ~**$2.6** at $0.308/h |
| stage-2 training to step 28,000 (10,390.7 s = 2.89 h) | ~**$0.9** |
| the job's other steps (checkout, deps, stage-1 no-op, provenance) | ~**$0.2** |
| **total** | **≈ $3.7–4.2** |

**What the cancel preserved, and it is most of the money.** The 16.24 GiB Z is on
`gpu-box-39184683`'s disk at `/opt/earth-cache/Z_actions_<whash>_<dhash>.npy` and the
re-dispatch reuses it, so the 8.5 h embed is **not** re-paid. Cancel stops the job and
leaves the disk; a destroy would have cost the embed, `run-415.pt` and the warm 4.5 GB
tensor. **DO NOT DESTROY vast 47724565.** The Z was never published to `embed-cache-v1`
(zero assets created since 2026-08-20), so that one disk is the only copy — which also
means `scripts/disk_hygiene.sh` will not touch it (tier 1 frees an embed cache only when a
release CONFIRMS a second copy; an unpublished `Z_*.npy` is tier 2, "never").

### 4 · §4 META-LESSON CANDIDATE — the run published the number that would have caught it

**Stage 2 had no gradient clipping and no monitor that would have flagged the divergence,
and the missing piece was not data — it was attention.** `stage2_grad_norm` was in every
record from step 2,000 onward. 8.24 → 787 is a 95× jump inside one log window and it sat on
the live branch for **hours** while the check-ins read `val_persistence`, `params_M` and
`train_windows` and pronounced the run healthy. Three things follow, and they are why this
is a §4 candidate rather than a bug report:

1. **A quantity that is published and not watched is not instrumentation.** #423's own
   trace contains its diagnosis in full. What was missing was a threshold — anything of the
   form "grad norm × 10 in one window ⇒ say so" — and a first-minutes checklist that ended
   at "the config is right" instead of continuing into "and the optimisation is sane".
2. **A default calibrated on one distribution is a hypothesis everywhere else.** Every one
   of the **8,080** stage-2 grad norms in the archive, across **83 runs**, is at
   `val_persistence` **3.09512** — the programme has, until now, exactly ONE stage-2
   z-space of experience, and both `--input-znoise 0.7` and "no clipping needed" are
   properties of it. §3b already says a level does not cross a tier boundary; #423 is the
   case where the level was not even in the dispatch, it was in the *absence* of a flag.
3. **The check that would have cost nothing is the one at the top.** §4.10 asks for the
   quantity that DISTINGUISHES the stories. A single grad norm sampled one step in 2,000
   cannot distinguish "one bad batch" from "every batch is now bad"; a window max and a
   clip-hit rate can, and they are free because clipping computes the norm every step
   anyway. Both are now logged.

### 5 · What changed in the code — `ml/temporal.py`, and the test that pins it

**`--grad-clip`, default `0.0` = OFF, and OFF MAKES NO CALL AT ALL** — not
`clip_grad_norm_(…, inf)`, not "a very large threshold". The default is the pre-2026-08-21
code path exactly, so **every archived monthly number stays bit-reproducible and no monthly
dispatch has to opt out of anything**. A negative value is refused at argument time (§0.3):
`clip_grad_norm_` would scale every gradient by a negative coefficient and walk uphill.

**Sizing 128.0 for the pentad arm, from the measured distribution.** Mined from
`ml-metrics` this session: **8,080** logged `stage2_grad_norm` values over **83** runs,
median **0.566**, p99 **4.279**, p99.9 **14.448**, **max 39.6165** (#308; #221 is 35.014).
How many of those 8,080 each candidate threshold would have clipped:

| threshold | 5 | 10 | 16 | 32 | 39.6165 | 64 | **128** |
|---|---|---|---|---|---|---|---|
| clipped | 34 | **16** | 5 | 2 | 0 | 0 | **0** |

**A clip at 10.0 is NOT a no-op at monthly** — it would have bound on 16 archived steps.
128.0 is **3.231×** the largest monthly norm ever recorded, **15.5×** the healthy pentad
norm (so it does not bind a healthy pentad run either), and **6.15×** below the smallest
pentad excursion. In distribution terms it is mild rather than exotic: 15.5 healthy norms
transferred onto the monthly median is a clip at 8.8, which bites ~0.25% of the archive's
steps — the ordinary regime for a transformer.

**New instrumentation, per §4.10, all free because the clip computes the norm anyway.**
`stage2_grad_norm` keeps reporting the **PRE**-clip norm (so the 83-run archive stays
comparable to it); `stage2_grad_norm_max`, `stage2_grad_clip_frac` and
`stage2_grad_nonfinite` report over the whole log **window** rather than the one sampled
step. The pair is what separates "healthy, never binds" (frac 0.0, max well under the
threshold) from "the clip is now setting the effective learning rate" (frac climbing off
zero) — and it says so a full window before a sampled norm could. A non-finite norm is
counted separately and kept out of the max, because `torch.maximum` PROPAGATES NaN and one
such step would otherwise pin every later `stage2_grad_norm_max` at NaN (§5.22).

**And the z-scale is now measured by the run instead of derived afterwards.**
`stage2_monitor` gains `z_rms` (the size of the z-space) beside `val_persistence` (the size
of its one-step change), and `input_znoise_sigma`, `input_znoise_rel_pers`,
`input_znoise_rel_zrms` — the sigma actually used beside the two scales it should be judged
against. #423's own znoise could not be settled from ANY record, only from a job log that
then expired; and this session had to derive the pentad z-scale indirectly from the codec's
stage-1 probes because the Z itself was unreachable. `input_znoise` and `grad_clip` are also
added to `stage2_config` and to the checkpoint's `args`.

**`tests/test_e044_grad_clip.py` — 5 checks, all passing.** Check 1 pins the mechanism with
EXACT expected values (§4.9) using the archive's own numbers: #426's 1.2439 through a clip
at 128.0 comes out `torch.equal` to its input, the archive's largest norm 39.6165 likewise,
and #423's 13,051.751 comes out at **127.999992** with cosine similarity 1.000000000000 to
its input direction. Check 3 is the archive-comparability proof: the pinned pre-fix revision
**877ae5b** and the working tree run the same toy at the same seed with `--grad-clip` at its
default, and all 31 parameter tensors compare `torch.equal`, with the val curve, the
grad-norm curve and `z_t+1` identical over 100 logged points. Check 2 adds the stronger
form — a clip at a threshold nothing reaches (`1e9`) is **also** bit-identical, because
`clip_grad_norm_` clamps its coefficient at 1.0 and multiplying by exactly 1.0 is exact —
and then asserts the EFFECT where it does bind (31/31 tensors moved) rather than the
invocation.

**NOT done, deliberately.** No `--input-znoise-rel` flag. The mis-scaling in §2(b) is real
and the relative form is the right eventual fix (§4.2, normalise by properties of the DATA),
but this dispatch does not use it, an unused knob is an untested knob, and §1's rule is that
a setting must not appear to apply and quietly do nothing. It belongs to the arm that
actually varies znoise at pentad. The reporting — which is what makes the defect visible in
the artefact — shipped.

---

<a id="fleet-2026-08-21-0138"></a>
## OPERATIONS · The fleet at 2026-08-21 01:38Z — three boxes, three jobs, no idle burn

Not an experiment. Recorded because it sets what the next check-in inherits.

| box | $/h | job | state at 01:38Z |
|---|---|---|---|
| `gpu-box-31479844` (vast 47724559, Quebec) | 0.294 | **#426** (E-043b-SEED1) at this hour; **#428 / #429 now** | Reading at 01:38Z: stage 2 step **4,000 / 200,000**, gpu **83%**, disk 58%. Was idle from #425's finish (00:39Z) to 01:12Z — ~33 min, ≈$0.16 of idle burn, closed by this dispatch. **SINCE LANDED: #426 (E-043b-SEED1) completed 16:59:11Z, 15 h 46 min, ≈$4.64**, and the box now carries **[#428 (HEADPUB e043b-xl144-nolonhold-s1) and #429 (E-043b-SEED1-roll)](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043b-seed1-roll)** |
| `gpu-box-39184683` (vast 47724565, Hong Kong) | 0.308 | **#423** (E-044) | **the embed→stage-2 TRANSITION HAPPENED at the 01:42:20Z push** — the ~8.5 h pentad embed finished (last frame 95.6%, month 3,003 / 3,142, 30,573 s) and `stage2_config` + `stage2_monitor` are on the branch. See E-044 §6 items 5, 6 and 10, now **PASS**: `params_M` **206.536**, `d_z` **32**, `train_windows` **251,337,502** (the pre-registered estimate, exact to the unit), `codec_holdout_lon` **`0,0`** and `train_lon_hold` **`none`**. `stage2_monitor.val_persistence` **21.44622** — finite and positive; it is **not** comparable to the monthly arm's 3.095 because it is a different codec's z-space, and its only job is to be the denominator of `stage2_val_zmse`, so read the RATIO at the next check-in, not the level. No `stage2_step` row yet (`log_every` = 2,000). **Vast's GPU stats read all-zero for this box**, which `fleet_health` flags as a blind CPU-BOUND check; the live branch is advancing normally, so the run is alive and the TELEMETRY is what is stale |
| `gpu-box-46996216` (vast 47913006, Austria) | 0.333 | **#419** (E-043f) | still inside step 16 `Train`, gpu **89%**, job wall **31 h 45 min / 2,600 min** |

**No box was stopped, because there was none to stop**: `fleet_health` reads **0 runners
online+idle** after the dispatch. §7's idle-burn rule is satisfied by occupancy, not by a
stop.

**Money.** Credit **$43.14**, balance $0.0000, burn **$1.2576/h** ⇒ runway **34.3 h**, i.e.
to **~12:00Z on 2026-08-22**. The burn decomposes as **$0.935/h of compute on the three
running boxes** and **$0.322/h of storage on nine STOPPED ones** — a quarter of the burn buys
no computation at all, and stopped-box storage is the line item most worth a sweep the next
time a session has slack.

| what is outstanding | needs | finishes |
|---|---|---|
| **#419** tail + probe ladder + upload | ~1 h + ladder, ~$0.5 | training ~01:20Z (arithmetic); ladder unknown — the daily ladder is untested |
| ~~**#426** (E-043b-SEED1)~~ **DONE** | estimated ~15 h, ~**$4.4**; actual **15 h 46 min, ≈$4.64** | **completed 16:59:11Z on 2026-08-21** |
| **#423** (E-044) stage 2 after the embed | ~20 h, ~**$6.2** | ~21:45Z on 2026-08-21 |
| **#426**'s headpub + roll (§5 of that entry) — dispatched as **[#428 + #429 (E-043b-SEED1-roll)](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043b-seed1-roll)** | ~3.7 h, ~**$1.1** — #428 done in ~6 min ≈$0.03, #429 ~3.5 h ≈$1.03 | **#429 in flight, ≈21:40Z on 2026-08-21** |
| the pentad `sroll:` over #423's head | ~14 h, ~$4.3 | ~12:00Z on 2026-08-22 |

**Everything but the last row fits inside the runway; the pentad sroll ends within an hour of
it.** Per §0e this is arithmetic reported early, not a reason to decline anything — nothing
was scaled down to fit.

---

<a id="e-044"></a>
## E-044 · #423 — DISPATCHED 2026-08-20 15:10:05Z. The first stage-2 head at any cadence but monthly

Written **at dispatch**, hypothesis and falsifier first, so the log cannot be rewritten to
fit the answer (§1). Governed by `claude/E044-pentad-stage2-spec.md`; the finished input set
and the artefact read-out behind `steps` are in `claude/E044-dispatch-READY.md`.

**E-044 · Trains a 206.5M xl144+znoise stage-2 head on the PENTAD tensor over #415's frozen
no-longitude-holdout pentad codec, with the stage-2 training pool open to every longitude and
years-only holdout (the first stage-2 run at any cadence but monthly) · params 206.5M head
over a frozen 37.976M codec · stage `stage-2` · data `family4_na025_pentad_r2` (C 40,
T 3,142, sha256 `37e146384b6f…`) · arch head 1024×16, K 24, stencil 145, ring
`spiral:111,4444,0.71,0.5`, znoise 0.7; codec 512×12, 4 heads, d_dec 256, d_z 32, patch 1 ·
steps×batch stage-1 **0** (resumes at its own recorded step 197,428 — nothing trains) /
stage-2 **200,000 × 256** · resume `run-415`.**

[#423 (E-044 pentad xl144+znoise stage-2 head) — the live status page](https://blauewelt.github.io/earth/status.html#run-423)

[#423 — the CI log](https://github.com/blauewelt/earth/actions/runs/32384499101)

[E-043e · #415, the codec this freezes](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043e)

[E-044 · the dispatch spec (project doc)](https://blauewelt.github.io/earth/docs.html?f=claude/E044-pentad-stage2-spec.md)

#### 1 · Hypothesis and falsifier

**HYPOTHESIS.** At matched stage-2 architecture, the pentad embedding forecasts better one
step ahead than the monthly one, because five-day bins carry the mesoscale evolution that
monthly averaging removes — and the stage-2 head has **6.5× the training windows** (~251M
against the monthly arm's ~38M) to learn it from.

**FALSIFIER, stated before the number exists.** If `z_t+1.mse_model / mse_persistence` does
not beat the monthly xl144+znoise arm's ratio, the extra temporal resolution buys nothing at
stage 2, and the pentad programme's cost — a ~10 h embed, a 16.24 GiB Z, a 4.5–6× more
expensive roll — is not justified by its forecast.

**Two reasons the headline is one-step MSE and not corridor AUC**, both from the spec: the
corridor AUC comes from a **separate** `sroll:` dispatch (§7a/§7b — `--horizon 73`, 3 starts,
day-defined bands, `horizon_auc_daymatched`), and at pentad that roll is **UNCERTIFIED by
construction** because `GATE_REF_BY_CADENCE` has no pentad entry and `e017_u1_s0`'s 0.643 was
measured over the monthly family-3 axis. The first published pentad roll is what would
ESTABLISH a pentad reference.

**SEEDS — §3b's one-seed licence does not reach here.** That licence is granted only for
rolled corridor AUC at the xl tier on the frozen f3 anchor and the monthly `family3_na025`
tensor. This run changes the **codec, the tensor and the cadence at once** and its headline
metric is not corridor AUC, so §3b's harder clause governs verbatim: *"any new metric,
cadence, tensor, codec or scale tier with no measured pair — the first result at a tier buys
its own replication."* Seed 0 runs now; **a `seed:1` arm is required before any number from
this run is written as a level rather than a first reading**, and it is cheap once the Z
cache is published because it skips the ~10 h embed.

#### 2 · `steps` is **197,428**, and this is the field that would have cost a day

The spec's §3 expected 200,000 on the reasoning that #415 carried `max_minutes 0`. **It
carried `max_minutes: "1150"`** and was re-fit fifteen times (E-043e §1). The value was read
off the `pixelmae-415` artifact — `step` **197428**, `args['steps']` **197428**, `tag`
`run-415`, `holdout_lon` **`'0,0'`**, 512×12/4/256, `d_z` 32, `patch` 1,
`chan_emb.weight` **(40, 512)**. A dispatch stating 200,000 would have trained **2,572**
stage-1 steps, changed the codec weight hash, and built a **different Z under a different
cache key** — a job that looks perfect and is not the experiment.

`max_minutes: "0"` on THIS dispatch is the other half of the same lesson: a non-zero budget
here would re-fit **stage 2's** schedule. `job_timeout: "2400"` (40 h) is the cap that
actually stops the job, against an estimated ~30 h.

#### 3 · Read-outs, decided in advance

- **HEADLINE** — `z_t+1.mse_model` / `mse_persistence`, plus the `stage2_val_zmse` curve on
  the fixed held-out monitoring batch. Both z-space, both keyed on held-out YEARS only,
  neither pools anything spatially.
- **SECONDARY, on the CODEC** — `probe_head.json` + `probe_head_raw3x3.json`, from
  `head_probe: true` (pinned in the recipe). **This also closes #415's own gap**: #414, #415
  and #416 each lost a head number to a copied `head_probe: "false"`, and this is the run
  that finally takes it at pentad. The references it will be judged against, written down
  now: **anchor head 0.691 [0.631, 0.746] (#406) · #386's own r1 pentad head 0.680
  [0.617, 0.740] (#409) · raw-3×3 control 0.683 [0.620, 0.742] · wind-only 0.670
  [0.601, 0.733]**.
- **EMITTED, LABELLED, EXCLUDED FROM EVERY CLAIM** — `rapid_probe`, `rapid_probe_kfold` and
  the in-training `stage2_probe` are all `hid[:, -1].mean(0)`, section-pooled, the read-out
  §3 distrusts at this cadence. They are kept because 95 archived bundles carry them and
  their bit-for-bit reproducibility is the protocol-determinism certificate, and they cost
  ~2.5 h of this job. Not disabled; not quoted.

#### 4 · The box, and the two hours this dispatch spent blocked

`gpu-box-39184683` (vast **47724565**, Hong Kong, RTX 4090, **504 GB RAM**, 100 GB disk,
$0.308/h) — #415's own box, holding `/opt/earth-cache/ckpt/run-415.pt` and the **only copy of
`family4_na025_pentad_r2` that exists anywhere**. It was `exited` and its host had no free
GPU: every start from **14:31Z** returned `resources_unavailable` (the #407 failure), and it
came up on the **28th consecutive attempt at 15:07:23Z**. The runner registered `online`,
`busy: false`, and #423 went out at 15:10:05Z. Full record of the block and of why no other
box was substituted — the pentad tensor is published nowhere, and a box that builds its own
tensor is the **E-008 box effect**, 0.041 on the head k-fold at a fixed seed — is at
[OPERATIONS · 2026-08-20 ~14:45Z](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#ops-2026-08-20-1445).

**504 GB RAM matters more than it looks.** `temporal.py` keeps the tensor eager:
`np.nan_to_num(X)` copies 34 GB beside the 34 GB `X` and `np.isfinite(X)` adds a 17 GB bool
— an **~85 GB peak** at `temporal.py:1265-1266` before `del X`, settling to ~51 GB, plus
~4.0 GB of int64 window-pool indices and the 17.4 GB Z in page cache. `probe_kfold.py` has
had the LazyPixels treatment since #388; `temporal.py` has not. On a 126 GB box that peak is
the single most likely way this run dies, and it dies as exit code 137 with no message. On
this one there is room.

#### 5 · Cost, and the runway it does not fit inside

| item | value |
|---|---|
| ESTIMATE, total | ~**30 h** = embed ~10 + train 200,000 × 0.27 s = 15.0 + pooled probes ~2.5 + ladder ~2.5 |
| money | ~**$9.2** at $0.308/h |
| `job_timeout` | 2,400 min (40 h) — an INPUT, not a cap on a self-hosted runner |
| Z | 3,142 × 86,698 × 32 × 2 B = **16.24 GiB**, 11 chunks at `embed_cache_sync`'s 1.5 GiB |
| credit at dispatch (15:10Z) | ~$5.6, burning $0.95–1.02/h with two boxes ⇒ exhausted ~20:20–20:45Z |
| **credit at 16:02Z — A TOP-UP LANDED** | **$54.48**, burn **$0.991/h**, runway **55.0 h** ⇒ ~23:00Z on **2026-08-22** |

**This run was dispatched over a runway it did not fit inside, per §0e** (*"please don't
worry about top ups and proceed in spite of remaining budget"*) — at 15:10Z it would have
died **inside its embed pass, ~5 h in, having produced nothing**, since the embed alone is
~10 h. **At 16:02Z the credit read $54.48 against $5.63 at 14:50Z: Chris topped up, and the
constraint is gone.** At $0.991/h the fleet now has **55 hours**, and everything outstanding
fits inside it with room:

| | needs | finishes |
|---|---|---|
| **#419** (E-043f daily codec) tail | ~11 h, ~$3.7 | ~02:50Z on 2026-08-21 |
| **#423** (E-044, this run) | ~30 h, ~$9.2 | ~21:45Z on 2026-08-21 |
| the follow-on pentad `sroll:` over the head #423 publishes (spec §7b: `--horizon 73`, 3 starts) | ~14 h, ~$4.3 | ~12:00Z on 2026-08-22 |
| **total against $54.48 at $0.991/h** | **~$17 of GPU + ~$14 of storage** | **fits, with ~$23 spare** |

**So §0e paid.** The dispatch went out at 15:10Z against $5.63 and a five-hour runway, and
52 minutes later it had 55 hours. A session that had "paused the wave to be safe" would have
cost the night and produced nothing, which is the failure that rule exists to forbid.

#### 6 · Verification — first minutes (§2, spec §5)

Recorded as they are read. Items marked **PENDING** were not yet decidable when this entry
was written at ~15:20Z.

State at **16:05Z**. **The job log is not readable while the job runs** (`/actions/jobs/<id>/logs`
returns 404 until it ends), so items whose evidence is a log LINE are marked
`PENDING — LOG` and the ones that could be settled from behaviour, the API or the live branch
are settled. Nothing was marked passed on an inference dressed as a reading.

| # | check | state |
|---|---|---|
| 1 | `runner_name` is `gpu-box-39184683`, not `gpu` | **PASS** — the jobs API reads `gpu-box-39184683` |
| 2 | Resolve prints the whole `RECIPE_*` block | **PASS (step)**, PENDING — LOG for the block itself. Step 5 `Resolve recipe` completed `success`; the identical block was reproduced locally by `bash scripts/resolve_recipe.sh` before dispatch |
| 3 | **`RESUMED … at step 197428`** then **`checkpoint is already at/past --steps; nothing to do`**. `training on to <N>` ⇒ **CANCEL WITHIN SECONDS** | **PASS — read off the run's OWN provenance, not inferred.** The status page's config line, rendered from `provenance.json`, reads **`steps×batch 197,428 × 512 · resume run-415 — loaded run-415.pt at step 197,428`**. `steps` and the checkpoint's recorded step are the same number, so `while s < a.steps` never turns over. Corroborated by behaviour: step 16 `Train` ran **15:42:25Z → 15:55:29Z = 13 min 4 s** and completed `success`, and `ml-live-423` carries **`phase.json` only — no `metrics.jsonl` at all**. Stage 1 training even the 2,572 steps a wrong `steps` would have bought writes metric rows and takes longer than the whole step did; training toward 200,000 would still be running nineteen hours from now. The literal `nothing to do` line is PENDING — LOG |
| 4 | `held-out months 219/3142 · NO lon holdout — all 481 cols train (--holdout-lon '0,0') · ocean 86698` | PENDING — LOG |
| 5 | `lon holdout · statistics (codec '0,0'): 0/481 cols · training pool (--train-lon-hold 'none'): 0/481 cols` — **both zeros** | **PASS, 2026-08-21 01:42Z** — settled off the live branch rather than the log: `stage2_config` carries **`"codec_holdout_lon": "0,0"`** and **`"train_lon_hold": "none"`**. Both holdouts are off, on both sides |
| 6 | `train windows: ~251,337,502` (ESTIMATE) — ~251M, not ~38M | **PASS, 2026-08-21 01:42Z, and the estimate was EXACT** — `stage2_config` reads **`"train_windows": 251337502`**, equal to the pre-registered figure to the unit. 6.5× the monthly arm's 38,488,680, which is the hypothesis's own premise |
| 7 | `embed cache needs 16.24 GiB`, branch taken = **disk** | PENDING — LOG, but the precondition holds: `fleet_health` reads **disk 57%** of 100 GB at 16:00Z, i.e. **~43 GB free** against `_cache_plan`'s `need + min(RESERVE, need)` on a 16.24 GiB need. The RAM branch is not expected |
| 8 | record `codec <whash> · tensor <dhash>` — the Z cache key; the §7a sroll refuses on disagreement | PENDING — LOG |
| 9 | embed ETA ~9–10 h (ESTIMATE); if ≫20 h check `gpu_util` | **the CPU-embed failure mode is RULED OUT**: `fleet_health` at 16:00Z reads **`gpu=98.999908%`, `cpu=6%`** on `gpu-box-39184683`, so the embed is on the GPU. Four eval scripts have silently embedded on CPU before; this one is not. The ETA line itself is PENDING — LOG |
| 10 | `stage-2 head on cuda (206.536M params)` — exactly **206,535,712**; 211.35M ⇒ d_z 64 ⇒ **cancel** | **PASS, 2026-08-21 01:42Z** — `stage2_config` reads **`"params_M": 206.536`** with **`"d_z": 32`**, `d_model` 1024, `layers` 16, `K` 24, `stencil` 145, `ring_km` `spiral:111,4444,0.71,0.5`, `seed` 0, `steps` 200000, `batch` 256. The cancel condition did not fire. (`input_znoise` is not a field this line prints, so the recipe's 0.7 is still PENDING — LOG) |
| 11 | first `stage2_lr` ~1e-3 and **not** 0.0 | PENDING — the first `stage2_step` row lands at step 2,000 (`log_every` = `steps // 100`), not at step 1 |
| 12 | ~0.27 s/step (ESTIMATE); above ~0.4 ⇒ re-time the budget and say so | PENDING — same reason as 11 |

**Timeline so far.** Dispatched 15:10:05Z · `Set up job` 15:09:50Z · **`Rescue an orphaned
checkpoint` 15:10:08Z → 15:39Z — twenty-nine minutes**, uploading `rescued-orphan-latest-423.pt`
(455,915,837 B, which is #415's own checkpoint and already published by hand as
`run-415__pixelmae.pt`) and `rescued-orphan-temporal-latest-423.pt` (1,076,218,089 B, a stale
leftover) from a Hong Kong box at ~0.5 MB/s · checkout → build 15:39–15:42Z (tensor cached,
seconds) · `Train` 15:42:25→15:55:29Z · `Upload checkpoint + eval` 40 s · **`Probes (K-sweep +
stage 2)` from 15:56:11Z**, `phase.json` `probes and stage 2`, GPU 99%. The status page at
16:04Z renders #423 under its **E-044** tag with the full `doc` string, the config line above
and the planned-schedule curve from `plan-423.json`; **PAGE ERRORS: none**.

**Operational finding worth a follow-up, recorded not fixed:** the orphan-rescue step spent
**29 minutes and ~$0.15 of GPU** re-uploading 1.5 GB that was already published or already
stale, before the job did anything. It runs before checkout, so it cannot consult the
release to see that `run-415__pixelmae.pt` is already there. On a fast-uplink box this is
invisible; on this one it is 5% of the embed pass.

Pre-dispatch checks that WERE completed, where the inputs are all they cost (§0.3): all
**25/25** input names matched against `.github/workflows/ml-train.yml`'s own
`workflow_dispatch.inputs` block, no extras and no omissions · `python3
tests/test_workflow_config.py` **5/5** · JSON **4,672 characters** against the 21,000-char
ceiling that took every dispatch down on 2026-08-17 · the recipe resolving to spec §1's exact
`RECIPE_*` block · and the resume checkpoint **opened and read** rather than assumed
(§0.1, §2 above).

---

<a id="e-043"></a>
## E-043 · Retire the 45°W–25°W longitude holdout — ALL FIVE ARMS HAVE NOW LANDED (#416, #414+#422, #417+#418, #415, #419); arm F's verdict is PROVISIONAL on a distrusted read-out, and arm B is down to its seed pair (#426)

The wave that follows from
[§0d · the skill map's central band is the held-out longitude block](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#holdout-lon-band-2026-08-19).
Five arms were planned; A, B, D and E went out at main `8f0e5141` at ~16:49Z and F
followed at 17:25Z. As of **2026-08-21 05:22:58Z all five have landed** — F last, after
35.96 h. Its verdict is **provisional**: it fired its own falsifier, on a read-out `ml/CLAUDE.md`
§3 distrusts at daily cadence, at n = 1
([E-043f · #419](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043f) §3). Arm B's roll
(#422) fired its own pre-registered falsifier by 19× — and arrived with a lead-time profile
that no other head in the archive has, so its headline number is recorded as **not yet
readable as a level**; that is [E-043b · the roll §7](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043b-roll)
and it is the most important open item on the wave.

**Two diagnostic runs hung off arm B; BOTH have now landed, and between them they have
narrowed the question to one word — `n`.**
**[E-043b-CONTROL (#424, re-roll of a KNOWN head on the new code)](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043b-control)**
**LANDED 2026-08-20 19:53:47Z and exonerated the evaluator**: `e032xl_u1_s0` came back at
corridor **0.68067 / 0.86700 / 0.22058**, reproducing #418's old-code record to every digit
it stores and decaying normally — the clean branch of a pre-registered two-way falsifier — so
#422's flat 0.939 belongs to **the head**, not to `ml/rollout_spatial.py`.
**[E-043b-MILESTONE (#425, #414's own step-600 head rolled)](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043b-milestone)**
**LANDED 2026-08-21 00:39:08Z and killed the structural explanation**: the step-600
milestone — identical geometry, identical seed, identical scope split, identical eval path,
600 optimiser steps instead of 200,000 — rolls corridor **0.02125 blended / 0.02967
`_trainlon`** with a **monotone** twelve-month decay **0.179 → −0.068** (spread 0.247, and
`acc` 0.419 → 0.141). **Branch (2) of the pre-registered two-way prediction fired**: *"if it
decays normally … the flat-and-inflated property was ACQUIRED over 200,000 steps of
all-longitude training and may be a real result."* Branch (1) — flat and near 0.94, spread
under ~0.05, STRUCTURAL, #422's number dead — did not fire and was not close.

**The epistemic position after both, stated carefully.** The instrument is clean (#424,
byte-identical reproduction). The structural story is dead (#425). Two of §7's four anomalous
signatures have **softened under scrutiny**: the 39-channel gain is **not** uniform (#422 −
#424 at h = 12 spans +0.055 to +0.467, sd 0.105 across 39 channels), and the "unforecastable
wind stress" argument does not separate the heads because **#424's own control reads `tau_x`
0.810 at h = 12** on the old code. The two that stand unchanged are the flat lead-time
profile itself and the complete absence of movement in any transport read-out. **What is left
between #422's 0.93933 and the paper is therefore not a code question and not a mechanism
question — it is §3b: n = 1 at a configuration no other head shares, on a number that would
headline. The seed pair is mandatory**, and it went out at 01:12:28Z as
**[E-043b-SEED1 (#426)](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043b-seed1)**.

| arm | run | what it is |
|---|---|---|
| A | **#416** (E-043a: monthly f3 codec retrained with NO longitude holdout) | **landed — this entry** |
| B | **#414** (E-043b: xl144 stage-2 head trained on an all-longitude pool over the EXISTING frozen anchor) → **#420** (HEADPUB) → **#422** (the roll) | **COMPLETE, AND THE ONE TO READ CAREFULLY — [E-043b (#414, training half)](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043b)** and **[E-043b · the roll (#420 + #422)](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043b-roll)**. Gate PASSED (0.643, 20th reproduction). The hole is **gone** — corridor `_trainlon` − `_holdlon` **0.0065** against the control pair's 0.646 / 0.667 — which is arm B's hypothesis confirmed. The corridor `_trainlon` figure **0.93933 vs the control pair's 0.86754** is **NOT** yet quotable: §7 of that entry records a flat lead-time profile, a 39-channel gain including wind stress, and **no movement at all in any transport read-out**. (#421 was the same roll and is VOID.) **Both diagnostics have landed: [#424 (E-043b-CONTROL)](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043b-control) cleared the roll code byte for byte, and [#425 (E-043b-MILESTONE)](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043b-milestone) killed the structural explanation — the step-600 head rolls 0.02967 `_trainlon` with a monotone 0.179 → −0.068 decay, so the property was ACQUIRED over 200k steps.** The arm's only remaining gap was n = 1, and **[#426 (E-043b-SEED1, the seed-1 xl144 all-longitude head)](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043b-seed1) has LANDED** — completed 2026-08-21 16:59:11Z after 15 h 46 min, ≈$4.64. **Its one-step forecast ratio REPRODUCED: 0.01400 against #414's 0.01392, pair \|Δ\| 0.00008**, so the training-half anomaly is a property of the configuration and not a single draw — but that ratio carries the same scope confound in both seeds and is still not a skill result. **The corridor verdict is [#429 (E-043b-SEED1-roll, the seed-1 xl144 head rolled beside the e017_u1_s0 gate)](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043b-seed1-roll), in flight, ≈21:40Z**, its headpub [#428 (HEADPUB e043b-xl144-nolonhold-s1)](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043b-seed1-roll) already complete — until #429's `_trainlon` \|Δ\| against #422's 0.93933 is in, no level from this configuration may be quoted. (§7(c)'s "uniform" is retired by #425 §6.4: the gain spans +0.055 to +0.467 at h = 12.) |
| D | **#417 / #418** (E-043d: sroll re-rolls, `_trainlon` / `_holdlon` split) | **COMPLETE — [E-043d1 (#417, xl233 pair)](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043d1)** and **[E-043d2 (#418, xl144 pair)](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043d2)** |
| E | **#415** (E-043e: fresh 38.0M pentad r2 codec, all longitude columns) | **LANDED — [E-043e (#415)](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043e)**, 12:55:52Z, recorded step **197,428** (the cosine was re-fit fifteen times despite `max_minutes 1150`), codec **published** as `run-415__pixelmae.pt`. **No head probe** — `head_probe: "false"` copied from #386, the third such miss in this wave — so its verdict arrives with E-044's ladder |
| F | **#419** (E-043f: fresh 38.0M DAILY codec on all 481 longitude columns) | **LANDED 2026-08-21 05:22:58Z after 35.96 h — RESOLVED, PROVISIONALLY: [E-043f (#419)](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043f)**. The first daily codec in the programme's history to reach a probe ladder at all. **Its pre-registered falsifier FIRED on both halves** — final `linear_r_deseas` **0.570** inside the 0.558–0.598 band, and the pooled trace flat across 40 probes from step 7,500 (0.545–0.596, slope −0.006 per 100k steps) — and the control comparison is the plain form of it: #410 (E-038c daily codec, the withheld-block run at this exact architecture) read **0.582** at step 60,000 where #419 reads **0.576**, so **returning the withheld quarter of the ocean bought the daily arm nothing**. **But read §3 of that entry before quoting any of this:** `head_probe: "false"` again (the fourth miss in this wave), so there is **no unpooled read-out** and every number here is one §3 distrusts at daily cadence; at n = 1, §3b forbids reading it as a closure. The codec is **published** as `run-419__pixelmae.pt`, so the settling move is an **eval-only ladder with `head_probe: "true"` over it — no retraining**. Also on the record from this arm: the pooled k-fold ties a raw wind-stress ridge on RAPID (0.612 vs 0.607, overlapping CIs, no paired test) while clearly beating it on the Florida Current (0.364 vs 0.110); probes were **62.4% of the training loop's wall clock**; and the job outran GitHub's 24-hour token ceiling, so both archive steps failed 401 and **reported success** |

<a id="e-043a"></a>
### E-043a · #416 — RESULT, completed 2026-08-19 20:46:10Z

**E-043a · Trains a FRESH 40.7M monthly f3 codec on ALL 481 longitude columns: the
anchor's exact geometry with the −45..−25 block returned to the stage-1 training pool
(recipe `f3-anchor-41M-nolonhold`, `holdout_lon '0,0'`) · params 40.693M · stage
`encoder` · data `family3_na025` (C 39, T 516, tensor sha256 `adcbe700fb6e…`) · arch
576×10, 8 heads, d_dec 768, d_z 64, patch 3 · steps×batch 60,000 × 512 (cosine 3e-4 to
zero; `max_minutes 0`, so nothing refits) · resume EMPTY — a fresh codec, there is no
no-holdout parent to continue from.**

**Code.** #416 → `head_sha` `8f0e5141`, job `train` on `gpu-box-46045353` (vast 47717160).

[#416 (E-043a monthly f3 codec, no lon holdout) — the CI log](https://github.com/blauewelt/earth/actions/runs/32278112904)

[probes-416.json on ml-metrics](https://github.com/blauewelt/earth/blob/ml-metrics/probes-416.json)

**Scale (rule 6).** parameters **40.693 M** (log: `codec parameters: 40.69M`; the
anchor's geometry exactly, 40,692,849) · batch
**512** · steps **60,000** · data points **40,514,400 train pixels** (= 480 train months
× 84,405 ocean pixels), against the anchor's 30,376,800.

#### 1 · The regime DID change — the log says so verbatim

This is the only direct proof that the mechanism the whole wave shares actually took
effect, so it is quoted rather than inferred. From #416's job log, lines 1348 and 1355:

> `held-out months 36/516 · NO lon holdout — all 481 cols train (--holdout-lon '0,0') · ocean 84405`

> `train pixels 40,514,400 · held-out pixels 3,038,580`

**Both lines match the pre-registration character for character.** The expectation was
written down before the run finished, from #62's own log
(`held-out months 36/516 · held-out lon block 80/481 cols · ocean 84405` /
`train pixels 30,376,800 · held-out pixels 13,176,180`), and predicted exactly
`40,514,400` train pixels and `3,038,580` held-out pixels. Arithmetic:
40,514,400 = 480 × 84,405; 3,038,580 = 36 × 84,405; and
40,514,400 / 30,376,800 = **1.33333… = exactly 4/3**, the pre-registered ratio. The 80
withheld columns carried 21,120 of 84,405 ocean pixels = **25.02%**, which is the 25.0%
this log already records for the rolled window.

`--holdout-lon=0,0` reached the trainer: `RECIPE_HOLDOUT_LON=0,0` appears in every
Resolve/Train environment block in the log, and `[0,0)` is the empty half-open interval,
so the mask is bit-identical to "none" while still parsing in the twelve eval scripts
that `float()` the field.

#### 2 · What it returned, and the number it must be read against

`probe_kfold` (year-blocked, pooled ridge over `Z.mean(1)` on the 26.5°N section),
monthly `family3_na025`, log verbatim:

> `actions    d_z=64  rapid  k-fold r +0.613 [+0.493, +0.716]  (n=240) · RMSE 2.20 Sv (sigma 2.79) · 18mo-lowpass r +0.803 · wind-only +0.568`

| probe (pooled ridge, year-blocked k-fold) | r | 95% CI | n | RMSE Sv |
|---|---|---|---|---|
| **#416 — no-lon-holdout codec** | **0.613** | [0.493, 0.716] | 240 | 2.20 |
| **f3_anchor41M (run-62/run-63, tag run-80) — the CONTROL** | **0.627** | [0.503, 0.735] | 240 | 2.17 |
| wind-only bar, this tensor (identical in both) | 0.568 | [0.428, 0.696] | 240 | 2.29 |

**Where the control number comes from, and why it is 0.627 and not 0.631.** It is the
archived `probe_kfold` block in **`probes-140.json`**, and identically in every one of the
**95 archived bundles** from #140 to #360 that froze `!run-62,run-63` on
`family3_na025` — §3b's *protocol determinism*, not 95 replicates. `probes-140.json` is
cited because it is the earliest archived bundle whose `provenance.json` records
**tensor sha256 `adcbe700fb6e…`, the same bytes #416 trained and was scored on**. The
programme's *headline* anchor number, the 0.631 [0.513, 0.732] / 2.16 Sv in
[E-003](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-003) and in
`ml/LEADERBOARD.md`, comes from **`probes-116.json`**, whose provenance predates the
tensor-sha field and which E-008 attributes to the **box effect** (two boxes each built
their own `family3_na025`; §"A confound that is NOT float noise"). 0.627 is the
tensor-matched comparison and 0.631 is the published one; the difference between them,
**0.004, is a build-of-the-tensor artefact and is itself more than a quarter of the
effect being discussed here.**

Other targets: FC **0.278** [0.191, 0.366] (n=490) vs wind-only 0.121 — the one clear
gain over wind in the bundle; MOVE 0.138 [−0.004, 0.279] vs wind-only −0.376; OSNAP 0.017.
`probe_sequence` peaks at K=12 (r_deseas 0.569). `dip_check`: out-of-fold r 0.613, sign
agreement **68.8%**, 2009–10 dip **44.3%** captured (anchor 51.2%).

In-training light probe, for the trajectory only (labelled per rule 5, never a
head-line): step-0 untrained `linear r_des +0.269`; `@2000 +0.491` against **#62's
+0.476 at the same step**; end of run `@58000 +0.459`. Reconstruction
`rec 0.1512 → 0.0980` over the 60k steps. (#62's +0.476 is second-hand — it was read out
of #62's own log by the dispatching session for the first-minutes check; that log has
since aged out of the Actions API and is **not independently re-verifiable now**.)

#### 3 · Is the −0.014 anything? — the paired test CANNOT be run, and this is a code gap

§3 requires a **paired** comparison (`scripts/paired_probe.py`), because two probes
scored on the same 240 months and the same year-blocked folds share most of their error
and their overlapping CIs say nothing. **It could not be run, for a reason that is
structural rather than an archiving oversight:**

- `paired_probe.py` requires `pred`, `target_sv` and `years` in both files.
- **`ml/probe_kfold.py` never writes them.** It computes `pred` — `r, lo95, hi95, n,
  rmse, sigma, pred = kfold_r(...)` — and then emits only the summary block
  (`r_kfold_deseas`, `ci95`, `n`, `rmse_sv`, `sigma_sv`, `r_lowpass18`,
  `wind_only_baseline`). Only `ml/probe_head.py` dumps the per-month arrays (its
  lines 469–471).
- So **no pair of pooled-ridge k-fold results in this programme's history has ever been
  paired-testable**, archived or not — not #416 against the anchor, not any other two.
- And #416 has **no `probe_head.json` at all** (see §4), so the head route is closed too.

**FIX, cheap and not dispatched:** have `probe_kfold.py` dump `pred` / `target_sv` /
`years` alongside its summary — ~2 KB per target, the arrays already exist in memory at
the point of the `json.dump`. That single change makes every future codec comparison
paired-testable. It is the smallest lever in this entry.

**So the difference is stated against the noise scale instead, per §3b.** The gap is
**−0.014** (0.613 − 0.627), or −0.018 against the published 0.631.

- §3b's **closest measured analogue** is the *codec head probe* row: a **codec-seed
  pair** at **0.92M** parameters (patch24, #18 / #43) on the **1° global** tensor moved
  the **pooled ridge by 0.012** and the **attention head by 0.036**. −0.014 is
  **1.2× that pooled-ridge codec-seed delta**.
- **That extrapolation must be flagged.** The analogue is at **0.92M params on a 1°
  global tensor**; #416 is **40.7M on the 0.25° North Atlantic tensor** — 44× the
  parameters and a different tensor family. §3b's own words: a band "is warranted only
  where it was measured", and **no codec-seed pair has ever been measured at 40.7M**.
  The 0.012 is the only number in the record that is even the right KIND of quantity.
- Two other scales bracket it and both make −0.014 smaller still: the **box effect**
  moved this very number by **0.004** on the same protocol, and E-003's whole capacity
  result — 44× the parameters, 0.92M → 40.7M — was **+0.011** and was logged as a NULL.
- The RAPID k-fold's own instrument noise is the widest in the programme: n = 240 with
  **~9 effective DOF after the 18-month low-pass**, CI width ±0.11, and §3b's measured
  probe-scale spreads run **0.036–0.245**.

**Reading, per §3b's consistency form:** #416 is **consistent with the anchor at n = 1**
— a −0.014 difference on a pooled probe whose only comparable codec-seed delta is 0.012
and whose tensor-build artefact alone is 0.004. It is **not** evidence that removing the
holdout hurt the codec, and it is **not** evidence that it helped. **Nothing here is a
level and nothing here is quotable**: this is a new codec regime with no measured pair,
so §3b's second clause applies in full — a headline claim buys its replicate first.

Against the run's own pre-registered **falsifier** ("if its in-training linear r_deseas
and its later `probe_kfold` rapid r sit at or below the anchor's on the shared columns,
the holdout was buying real regularisation"): the probe r is **nominally below** and the
early light probe is **nominally above**. Split verdict on an instrument that cannot
resolve either sign — **the falsifier is not triggered and is not cleared.** The pooled
monthly probe is simply too blunt to be the arbiter of this question, which was the prior
going in and is why the wave does not rest on this arm.

#### 4 · What #416 does NOT test — read this before reading its number

**(a) It is the CODEC half only.** The holdout change has two halves, and this arm moves
the stage-1 one. The prior was that it would change **little**, and it was a measured
prior:
[E-019b1](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-019)
measured the retrained deep-T decoder at **1.43% rmse² on held-out longitudes against
0.85% on trained ones** (and 1.90% on held-out *months*) — the codec already generalised
across the block essentially perfectly, and the §0d finding's own conclusion was that the
skill band is *"entirely a stage-2 forecast-head generalisation failure."* **Expected
direction: ~zero, with a weak prior toward a small gain from 4/3 the training pixels.
Observed: ~zero (−0.014, inside every relevant noise scale). Consistent with the prior.**
A pooled 240-month ridge with ~9 effective DOF was never going to resolve "little", and
it did not.

**(b) THE DECISIVE ARM IS #414, NOT THIS ONE.**
**#414 (E-043b: xl144 stage-2 head trained on an all-longitude pool over the EXISTING
frozen anchor)** is the clean test of the holdout change: it resumes `run-62,run-63` at
step 60,000 with **stage 1 training nothing**, so the codec is the anchor, the **Z cache
is byte-identical**, and the *only* thing that differs is the stage-2 training pool. That
is where §0d located the failure and that is where the effect, if there is one, has to
show up — scored on rolled corridor AUC, the one metric §3b licenses at n = 1.
**Do not read #416 as the verdict on the longitude holdout.** #416's contribution is
narrower and still worth having: it proves the training-pool mechanism fires (§1) and it
shows the codec's own read-out did not move when it did.

**(c) #416 HAS NO HEAD NUMBER — recorded as a gap, with its cause.** The bundle carries
`probe_kfold.json`, `probe_sequence.json`, `dip_check.json`, `provenance.json` and the
archiver's own line `not present: … probe_head.json, probe_head_raw3x3.json …`. Cause:
`head_probe: "false"` in the inputs, because #416's dispatch inputs were copied verbatim
from **#62 (the anchor's own dispatch)**, which **predates the `head_probe` flag**. Per
§3 the pooled ridge "is the comparable-to-history number, never the verdict", so the
read-out this programme actually trusts was never taken on this codec.

> **NEXT CHEAP ACTION (recorded, NOT dispatched):** an **eval-only** re-dispatch against
> #416's published checkpoint with `head_probe: true` — no training, one probe ladder —
> would produce `probe_head.json` + `probe_head_raw3x3.json` for this codec and, because
> `probe_head.py` writes `pred`/`target_sv`/`years`, would also make it the **first
> no-holdout codec that can be paired-tested** against the anchor's head. #406's
> precedent prices an eval-only ladder at ~2 h on a ~$0.29/h box.

#### 5 · Cost (§3)

| item | value |
|---|---|
| box | `gpu-box-46045353` (vast 47717160) |
| wall clock, whole job | **3 h 56 min 37 s** (16:49:33Z → 20:46:10Z) |
| training loop | **13,683 s** elapsed at step 60,000 = 3.80 h (0.228 s/step wall, including the 29 interleaved light probes and 5 full probes; ~0.203 s/step of pure training) |
| dollars | **≈ $1.05** at ~$0.268/h |
| dead dispatches | **none** — #416 ran once, green, first try |

The $0.268/h is **inferred, not read off Vast**: the wave's five-box burn was $1.524/h at
dispatch and $1.256/h at 20:55Z after this box was stopped, and the only other change in
between was #419 starting on `gpu-box-46996216`, which was already burning idle at
$0.333/h and so contributes no delta. It is consistent with the ~$0.26–0.29/h this wave's
monthly boxes have billed. Treat the dollar figure as ±10%.

---

<a id="e-043d1"></a>
### E-043d1 · #417 — RESULT, completed 2026-08-19; the published corridor AUC is a floor, not a level

**E-043d1 · Re-rolls three ALREADY-PUBLISHED heads — the `e017_u1_s0` validation gate and
the E-035 xl233 clean seed pair — over the monthly family-3 tensor to read the new
`_trainlon` / `_holdlon` split beside every scope, decomposing each published aggregate
into the pixels the stage-2 head was trained on and the 45°W–25°W columns it never saw
(re-roll of the heads scored in #382/#394/#396/#413) · params 40.693M codec + 217.3M per
xl233 head (gate is a 32.0M 576×8 head) · stage sroll (eval-only — NOTHING trains; the step
count below is the checkpoint's own) · data `family3_na025` (T 516, C 39, sha256
`adcbe700fb6e…`) · arch codec 576×10, 8 heads, d_dec 768, d_z 64, patch 3; xl233 heads
1024×16, K 24, stencil 234, ring `spiral:111,4444,0.71,0.5` · steps×batch 60,000 × 512
(= `f3_anchor41M`'s own recorded step count; zero training steps) · resume
`!run-62,run-63` (`f3_anchor41M`, frozen)**

**Code.** #417 → `head_sha` `8f0e5141`, job `sroll:` on `gpu-box-31479844`, wall 24,146 s
of head time. Archive `probes-417.json` on `ml-metrics`; live progress records in
`run-417.jsonl` confirm `total: 714` steps per head, which is the protocol's own count.

[E-043 · the wave this arm belongs to](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043)

[§0d · the diagnosis this arm was dispatched to MEASURE rather than estimate](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#holdout-lon-band-2026-08-19)

#### 1 · The protocol certificate

`gate` block: head `s1_s0`, `got.auc` **0.643**, bands `h1-3` 0.47 / `h4-6` 0.375 /
`h7-12` 0.492, **`pass: true`, `fails: []`** — against `gate_ref` 0.643 with
`tol` 0.0101. That is the **nineteenth** reproduction of this number (#228 … #413 → #417).
Per §3b it is PROTOCOL determinism and **not** a replicate; it is what licenses reading
everything below as a measurement of the heads rather than of the box.

Roll configuration, read off the file: `horizon` 12, `hold_years` 2009 / 2017 / 2023,
`K` 24, `probe.val_tail_r` 0.606, `corridor_def` = 75.0th percentile of current speed,
threshold 0.1867, dilated 2 cells with a 3×3 square, union with the 26.5°N section,
**`n_px` 29,627 of 84,405**.

#### 2 · Pixel inventory per scope — what the split is a split OF

`holdout_lon` (`arg` `-45,-25`, `lo` −45.0, `hi` −25.0, rule
`(lons >= lo) & (lons < hi)`, **80 of 481 columns**, `excluded_from` both
`train.py`'s stage-1 pixel MAE and `temporal.py`'s stage-2 head pool):

| scope | total px | in-block (`_holdlon`) | trained (`_trainlon`) | in-block share |
|---|---|---|---|---|
| gate (600 ∪ section subset) | **864** | **229** | **635** | 0.2650 |
| AMOC corridor | **29,627** | **7,089** | **22,538** | 0.2393 |
| rolled window | **84,405** | **21,120** | **63,285** | 0.2502 |
| RAPID 26.5°N section | **265** | **80** | **185** | 0.3019 |

**The per-scope `n_px` field is NOT in this artefact and that is expected**, not a defect:
scope-level `n_px` landed in `9c1fbb0`, and #417 checked out `8f0e5141`, which is its
ancestor. The counts above are `holdout_lon.px.{scope}.{in_block, of, frac}` plus
`corridor_def.n_px`; `_trainlon` is `of − in_block`. The section row is emitted in the
inventory but **no head carries a `section` scope block**, so it cannot be scored here.

**The scored-ELEMENT share is not the pixel share, and the file says so.** Each
`chan_skill` row carries `n`, the count of finite (pixel × channel × start) values at that
horizon. At h = 1, corridor `n` = 37,528,668, of which `_trainlon` holds 27,694,368
(**73.80%**) and `_holdlon` 9,834,300 (**26.20%**) — and the two sum to the parent
**exactly**, at every horizon, in every scope. So the split is a genuine partition. But
73.80% ≠ the 76.07% pixel share, because the held-out block averages **38.54 finite
channels per pixel against 34.13 on the trained set** (39 possible) — it is deep open
ocean with full Argo coverage, where the trained set carries shelf pixels missing the deep
levels. Gate: 70.50% / 29.50%. Window: 70.81% / 29.19%. `n(h)` falls linearly as
`3 × (13 − h)` accumulations, which is the staggered-start protocol: **234 scored roll
steps per head** (78 per holdout year), plus 240 hindcast and 240 future.

#### 3 · The numbers

`horizon_auc` as stored (3 dp), and **recomputed to 5 dp from each block's twelve archived
per-horizon `msss_clim` values**, which is §3b's own method and the only form in which
these deltas are not rounding artefacts. `auc_damped` in the same order.

| head · scope | blended (as published) | `_trainlon` | `_holdlon` |
|---|---|---|---|
| **gate `s1_s0`** · corridor | 0.589 · **0.58908** | 0.804 · **0.80425** | 0.058 · **0.05767** |
| gate `s1_s0` · window | 0.622 · **0.62200** | 0.814 · **0.81367** | 0.120 · **0.12017** |
| gate `s1_s0` · gate scope | 0.643 · **0.64283** | 0.805 · **0.80533** | 0.154 · **0.15350** |
| **xl233 s0** · corridor | 0.675 · **0.67492** | 0.865 · **0.86525** | 0.206 · **0.20592** |
| xl233 s0 · window | 0.698 · **0.69775** | 0.855 · **0.85550** | 0.283 · **0.28342** |
| xl233 s0 · gate scope | 0.723 · **0.72250** | 0.842 · **0.84242** | 0.362 · **0.36175** |
| **xl233 s1** · corridor | 0.673 · **0.67292** | 0.866 · **0.86600** | 0.196 · **0.19633** |
| xl233 s1 · window | 0.696 · **0.69617** | 0.854 · **0.85425** | 0.282 · **0.28175** |
| xl233 s1 · gate scope | 0.720 · **0.71958** | 0.841 · **0.84067** | 0.355 · **0.35550** |

`auc_damped` (against the damped-persistence reference) moves the same way and by the same
amounts: corridor gate 0.57717 → 0.79975 / 0.03242; xl233 s0 0.66567 → 0.86200 / 0.18458.

**`amoc_bands`** (transport truefit r, unchanged by the split — the 26.5°N section has no
`_trainlon` block): gate `h1-3` 0.470 / `h4-6` 0.375 / `h7-12` 0.492 (n 99 / 72 / 63);
xl233 s0 0.502 / 0.416 / 0.512; xl233 s1 0.460 / 0.350 / 0.443. The seed spread on these is
**0.042 / 0.066 / 0.069** — §3b's "transport band r spreads 0.05–0.07 on the same
checkpoints where the corridor reproduces to 0.002", reproduced here for a third pair.

> **CORRECTED 2026-08-20 by [#418 / E-043d2 §7](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043d2).** The
> "0.05–0.07 regime" this paragraph confirmed did not exist: it was two replicate groups
> summarised to one number each. The archive was re-mined in full — **five pairs and one
> triple, per-band spreads 0.003 – 0.119, pooled sd 0.041 (15 dof)** — and §3b's row was
> rewritten. xl233's 0.042 / 0.066 / 0.069 is unchanged and is now the MIDDLE of the range,
> not a confirmation of its edges.

**`long`** (20-year hindcast from context end 2004-12): gate `r_trained` 0.774 (n 195) /
`r_heldout` 0.454 (n 36) / `r_lp18` 0.864 / `amp_lp18` 0.583; xl233 s0 0.770 / 0.478 /
0.864 / 0.768; xl233 s1 0.777 / 0.441 / 0.835 / 0.763. (These `trained`/`heldout` labels are
the YEAR holdout, not the longitude one.) `future` carries 240 rolled `sv_des` from
2024-12 for each head.

#### 4 · Every published corridor AUC understates skill on trained pixels — and the earlier ESTIMATES ARE RETIRED

The §0d diagnosis said the direction was deflationary and gave **reweighted estimates off
the h = 6 map**: `s1_s0` 0.589 → ≈0.75, xl89 0.674 → ≈0.82, xl144 0.681 → ≈0.82. Those were
arithmetic over one horizon of one map. **They are now superseded by measurement, and the
measurement is higher than the estimate in both cases it can be checked against:**

- gate `s1_s0` corridor: estimate ≈0.75 → **measured 0.80425** (+0.054)
- xl233 corridor (seed mean): the estimate's nearest sibling ≈0.82 → **measured 0.86563**
  (+0.046 against the xl89/xl144 estimate; xl233 itself was never estimated)

**The ≈0.75 / ≈0.82 figures are RETIRED.** Do not quote them again; quote the split. The
general statement they were reaching for survives and is now measured: **every corridor and
window AUC in `ml/LEADERBOARD.md`, in this log and in the paper is a LOWER BOUND on the
same head's skill over the pixels it was trained on**, by **+0.215** (gate) and **+0.192**
(xl233) on the corridor. Nothing in the archive is inflated; the ranking is unaffected,
because every arm is scored over the same pixels.

#### 5 · The stencil advantage survives, smaller — and a quarter of the published figure was hole-patching

Corridor, xl233 seed mean against the frozen 1-point gate, all five decimals:

| scope of the comparison | advantage |
|---|---|
| **blended — the published figure** | 0.67392 − 0.58908 = **+0.08483** |
| **`_trainlon` — honest pixels** | 0.86563 − 0.80425 = **+0.06138** |
| `_holdlon` — the training hole only | 0.20112 − 0.05767 = **+0.14346** |

**The share of the published advantage that is present on trained pixels is
0.06138 / 0.08483 = 72.3%; the remaining 27.7% is what the blend adds by including the
hole.** On the window scope the same computation gives +0.07496 blended, **+0.04121**
trainlon, +0.16242 holdlon — **55.0% / 45.0%**.

**How that share was computed, and what it is not.** It is a ratio of two AUC
DIFFERENCES, both taken on the same rolled fields at 5 dp. It **replaces** the §0d
table's 48% / 59% in-block shares, which were a *different quantity* — per-pixel MSSS at
the single horizon h = 6, decomposed by pixel share — and must not be read as the same
number. **`msss_clim` is `1 − Σmse_model / Σmse_clim`, a ratio of sums, so an AUC does not
decompose linearly in pixel share and this share is not a pixel-share attribution.**
Measured on this file, per horizon: the `n`-weighted linear blend of the two children
differs from the parent by **+0.009 … +0.031 on the corridor** (parent BELOW the linear
blend) and by **−0.046 … −0.019 on the gate scope (the extreme is the gate head's own −0.046; the xl233 heads span −0.028 … −0.019)** (parent ABOVE it) — the sign is
scope-dependent. Equivalently, the weight `w` that solves `parent = w·trainlon +
(1−w)·holdlon` is **0.697–0.723** on the corridor and **0.746–0.764** on the gate scope,
against `n`-shares of 0.738 and 0.705. So: quote 72.3% as *"the fraction of the published
corridor advantage that survives on trained pixels"*, and never as *"27.7% of the pixels
did the work"*.

**What survives is a real effect, by §3b's own bar.** +0.06138 on trained longitudes is
**2.5× the §3b threshold of 0.025** and **29× the xl-tier pooled seed sd of 0.0021**
(95% upper bound 0.0037 → 17×). Spatial coupling pays on honest pixels. And the
qualification the §0d entry attached to every corridor comparison is now **quantified
rather than asserted**: about a quarter of the published corridor figure, and about
half of the published window figure, is the stencil head reading its neighbours across a
hole in its own coordinate input. *"Spatial coupling helps forecasting"* and *"spatial
coupling helps you extrapolate into a training hole"* are both true, and the first is the
larger of the two on the corridor.

**One caveat that must travel with the +0.085 / +0.061 figures.** The gate is a **32.0M
576×8 stencil-1 head at 60k steps**; xl233 is a **217.3M 1024×16 stencil-234 head at 200k
steps**. That contrast is stencil AND capacity AND step budget, not stencil alone. It is
the comparison the archive has always drawn (the §0d table calls it *"stencil head vs the
frozen 1-point gate"*) and the split does not change its composition — but the clean,
capacity-matched stencil ladders are E-022's and E-035's (xl233 stencil 234 against xl144
stencil 145 at identical geometry), and those are the ones that isolate the stencil.
Splitting THOSE is #418's job.

#### 6 · The split is a stable measurement, not a noisy one — with one exception

The E-035 xl233 pair are two independently seeded stage-2 heads, so their spread on the new
scopes IS a seed spread (unlike the gate's nineteenth reproduction, which is protocol
determinism). One pair, 1 dof:

| scope | seed 0 | seed 1 | \|Δ\| |
|---|---|---|---|
| corridor **`_trainlon`** | 0.86525 | 0.86600 | **0.00075** |
| corridor blended | 0.67492 | 0.67292 | 0.00200 |
| corridor **`_holdlon`** | 0.20592 | 0.19633 | **0.00958** |
| window `_trainlon` | 0.85550 | 0.85425 | 0.00125 |
| window `_holdlon` | 0.28342 | 0.28175 | 0.00167 |
| gate-scope `_trainlon` | 0.84242 | 0.84067 | 0.00175 |
| gate-scope `_holdlon` | 0.36175 | 0.35550 | 0.00625 |

**`_trainlon` is the TIGHTEST reading of this pair anywhere in the archive** — 0.00075,
a third of the blended 0.00200 and well inside the xl tier's pooled sd of 0.0021. The
trained-longitude corridor AUC is therefore at least as readable an instrument as the
number it replaces, and §3b's one-seed licence should extend to it on the same terms.

**`_holdlon` is not.** 0.00958 on the corridor is **4.6× the tier's pooled sd** and larger
than the largest blended pair delta ever measured at this tier (0.0051, E-032 xl144).
Extrapolation into the hole is a genuinely less stable quantity than forecasting inside the
trained region, which is what one expects. **A held-out-longitude AUC must not be quoted at
n = 1**, and a `_holdlon` difference smaller than ~0.05 is not readable from one pair.

**Follow-up, not done here:** §3b's rule is that the spread table is extended in the same
commit as a new replicate. This pair adds a `_trainlon` row (|Δ| 0.00075) and a `_holdlon`
row (|Δ| 0.00958) at the xl tier, both n = 1 pair. `ml/CLAUDE.md` was outside this
session's ownership and was **not** edited; the rows are recorded here so the next session
can move them. **DONE 2026-08-20**: both rows are in §3b, and #418 took each of them to
**2 pairs** — `_trainlon` pooled sd 0.00066, `_holdlon` 0.01079.

#### 7 · What `_holdlon` 0.058 means for the gate — the mechanism, measured

The 1-point gate scores **0.05767** on corridor pixels inside the block, against 0.80425
outside it. That is not "degraded", it is **essentially nothing**: `msss_clim` near zero
means the rolled forecast is no better than climatology. A head with **no neighbours** has
literally no channel by which information from a trained column can reach an untrained
one — its only spatial input is `coords = [lat/90, lon/180]` (`temporal.py:826`, into
`static_ctx` at 1343/1351), and longitude is a **literal input feature with a 20° hole in
its training range**. Asked to forecast at a longitude it never saw, it has nothing to say,
and it says nothing. This is exactly what a coordinate-input hole predicts, and it is the
mechanism the paper's Figure 7 (`fig:gulfstream`) draws: the meridional band of collapsed
skill through the central Atlantic is the model's spatial generalisation gap, not an ocean
feature.

The stencil head is at **0.20112** in the same block — four times the gate, still far below
its own 0.86563 outside. Neighbours partially patch the hole and do not close it, which is
the smooth-ramp behaviour the §0d entry measured column by column (0.702 / 0.485 / 0.341 /
0.233 / 0.186 inward from 45°W). Both readings are consistent, and the split now measures
what the map only showed.

#### 8 · Cost (§3)

| item | value |
|---|---|
| box | `gpu-box-31479844` (shared with #418, by design) |
| gate head `s1_s0` | **1,983.0 s** (`wall_s`) |
| xl233 s0 | **11,082.9 s** |
| xl233 s1 | **11,080.2 s** |
| head time, total | **24,146.1 s = 6.71 h** |
| dead dispatches | **none** |

Against the same heads without the split: #394 read 1,948.8 / 10,750.9 / 10,754.1 s and
#413 read 1,986.1 / 11,019.0 s. **The six extra scope accumulations cost nothing
measurable** — the run-to-run scatter on a fixed head is ±2–3% and the deltas sit inside
it. Per-step rate, 714 steps per head (234 scored + 240 hindcast + 240 future):
**14.7–15.5 s/step** across #394 / #401 / #413 / #417, and a stencil-145 head (#401,
10,514.2 s) is only 2% cheaper than a stencil-234 one — the roll's cost is **not**
dominated by the stencil gather. That rate is the basis of E-044's pentad roll arithmetic.

#### 9 · What is still outstanding

**#418 (E-043d2: sroll re-roll of the E-032 xl144 pair + gate, `_trainlon` / `_holdlon`
split)** is the second half of arm D, queued behind #417 on the same box by design. **Its
splits are the ones Table 5 needs**: xl144 is the paper's flagship rung and the head whose
h = 6 map produced every estimate section 4 has just retired, and it is the
capacity-matched partner that turns the gate contrast of section 5 into a clean stencil
ladder. Nothing in Table 5 should be revised until #418 lands — the xl233 numbers here
establish the shape of the correction, not its value at the rung the paper quotes.

Also still outstanding, unchanged by this arm: **#414 (E-043b: xl144 stage-2 head trained
on an all-longitude pool over the existing frozen anchor)** remains the decisive arm. #417
measures how large the hole's effect on the reported numbers is; only #414 says what the
numbers become when the hole is not there.

---

<a id="e-043d2"></a>
### E-043d2 · #418 — RESULT, completed 2026-08-20 06:10:07Z; the flagship rung's split, and the width story survives onto honest pixels

**E-043d2 · RE-ROLLS the E-032 xl144 seed pair (`e032xl_u1_s0`, `e032xl_u1_s1`) plus the
frozen `e017_u1_s0` gate to read the `*_trainlon` / `*_holdlon` split beside every scope,
so the paper's FLAGSHIP corridor AUC can be decomposed into the 75.98% of corridor pixels
the heads trained on and the 24.02% they never saw (arm D2 of the E-043 wave) · params
40.693M codec (frozen) + 211.353M per head · stage `sroll` (eval-only — NOTHING trains;
the step field is the codec checkpoint's own count) · data `family3_na025` (C 39, T 516) ·
arch codec 576×10, 8 heads, d_dec 768, d_z 64, patch 3; heads 1024×16, K 24, stencil 145,
ring `spiral:111,4444,0.71,0.5` · steps×batch 60,000 × 512 (= f3_anchor41M's recorded step
count; zero training steps) · resume `!run-62,run-63` (f3_anchor41M, frozen).**

**Code.** #418 → `head_sha` `8f0e5141`, job `train` on `gpu-box-31479844` (vast 47724559),
23:39:13Z → 06:10:07Z.

[#418 (E-043d2 sroll re-roll, gate + xl144 pair) — the CI log](https://github.com/blauewelt/earth/actions/runs/32278157309)

[probes-418.json on ml-metrics](https://github.com/blauewelt/earth/blob/ml-metrics/probes-418.json)

[E-043d1 · #417, the xl233 half of the same arm](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043d1)

#### 1 · Integrity first — the run is readable

Three checks, all passed, all read out of the artefact rather than the run's exit code:

| check | expected | got |
|---|---|---|
| gate `e017_u1_s0` horizon AUC | 0.643 ± `GATE_TOL` 0.0101 | **0.643**, `gate.pass true`, `fails []` |
| gate bands h1-3 / h4-6 / h7-12 | 0.470 / 0.375 / 0.492 | **0.470 / 0.375 / 0.492** |
| xl144 blended corridor AUC | 0.68067 / 0.67558 as scored by #356 | **0.68067 / 0.67558**, recomputed to 5 dp from the twelve per-horizon `msss_clim` values |
| pixel inventory | corridor 29,627 px, 7,089 in-block | **identical to #417** — same `holdout_lon` block, same `corridor_def` |

The blended pair reproduces **exactly**, which is the point: nothing in the split changed
what the parent aggregate reads, so the `_trainlon` / `_holdlon` blocks are a genuine
decomposition of a number the log already published, not a re-scoring of it. §3b's warning
applies in full — this is **PROTOCOL determinism, the nineteenth reproduction of the gate,
not a replicate**, and none of these numbers may be entered in a spread column as such.

Scored-element partition, h = 1, corridor: `_trainlon` 27,694,368 + `_holdlon` 9,834,300 =
37,528,668 = the parent, **exactly** — as in #417, and at every horizon in every scope.

#### 2 · The numbers

`horizon_auc` as stored (3 dp) and **recomputed to 5 dp from each block's twelve archived
per-horizon `msss_clim` values** (§3b's method — at 3 dp the deltas below are rounding
artefacts). `auc_damped` recomputed the same way.

| head · scope | blended (as published) | `_trainlon` | `_holdlon` |
|---|---|---|---|
| **gate `s1_s0`** · corridor | 0.589 · **0.58908** | 0.804 · **0.80425** | 0.058 · **0.05767** |
| gate `s1_s0` · window | 0.622 · **0.62200** | 0.814 · **0.81367** | 0.120 · **0.12017** |
| gate `s1_s0` · gate scope | 0.643 · **0.64283** | 0.805 · **0.80533** | 0.154 · **0.15350** |
| **xl144 s0** · corridor | 0.681 · **0.68067** | 0.867 · **0.86700** | 0.221 · **0.22058** |
| xl144 s0 · window | 0.703 · **0.70300** | 0.858 · **0.85842** | 0.295 · **0.29508** |
| xl144 s0 · gate scope | 0.728 · **0.72775** | 0.846 · **0.84633** | 0.371 · **0.37083** |
| **xl144 s1** · corridor | 0.676 · **0.67558** | 0.868 · **0.86808** | 0.201 · **0.20125** |
| xl144 s1 · window | 0.699 · **0.69933** | 0.859 · **0.85892** | 0.283 · **0.28250** |
| xl144 s1 · gate scope | 0.722 · **0.72200** | 0.848 · **0.84783** | 0.344 · **0.34433** |

`auc_damped` moves the same way and by the same amounts: corridor xl144 s0
0.67142 → 0.86400 / 0.19950; s1 0.66625 → 0.86500 / 0.17975.

`long` (20-year hindcast from context end 2004-12): gate `r_trained` 0.774 (n 195) /
`r_heldout` 0.454 (n 36) / `r_lp18` 0.864 / `amp_lp18` 0.583; xl144 s0 0.780 / 0.417 /
0.852 / 0.739; xl144 s1 0.787 / 0.350 / 0.822 / 0.755. (`trained`/`heldout` here is the
YEAR holdout, not the longitude one.) `probe.val_tail_r` 0.606. `future` carries 240 rolled
`sv_des` per head from context end 2024-12.

#### 3 · THE RESULT · On trained pixels, xl144 and xl233 are indistinguishable — and xl144 is the one that is (very slightly) ahead

This is what arm D was for. The two top rungs of the width ladder, both now scored on the
pixels they were actually trained on:

| rung | seed 0 | seed 1 | pair mean | pair \|Δ\| |
|---|---|---|---|---|
| **xl144** (stencil 145, 211.353M) — #418 | **0.86700** | **0.86808** | **0.86754** | **0.00108** |
| **xl233** (stencil 234, 217.3M) — #417 | 0.86525 | 0.86600 | 0.86563 | 0.00075 |
| | | | **\|Δmeans\| 0.00192** | |

Against the blended scope, where the same four heads read 0.68067 / 0.67558 and 0.67492 /
0.67292 — pair means **0.67813** and **0.67392**, **\|Δmeans\| 0.00421**.

**Scoring only trained pixels HALVES the gap between the two rungs, from 0.0042 to
0.0019, and does not change its sign.** Both numbers are far below §3b's 0.025 bar; both
are below the tier's pooled blended seed sd (0.0021, 7 dof); and the sign is the wrong way
round for width — the NARROWER stencil is nominally ahead at both scopes.

Is 0.00192 resolvable? Honestly, no, but the margin is thinner than the blended reading
suggested and it is worth stating exactly. The `_trainlon` scope has its own, much tighter
pair noise: pooled over #417's and #418's pairs, sd **0.00066** on 2 dof. A difference of
two two-seed means has that same sd, so 0.00192 is **2.9σ** — inside a 2-dof *t* 95%
interval of ±0.00283, and therefore **not resolvable**, but only just. The correct statement
is a consistency, not a level (§3b):

- ✅ **xl233's trained-pixel corridor AUC is consistent with xl144's; the difference is
  0.0019 ± (2 dof), inside the interval, and if anything favours xl144.**
- ❌ *xl144 beats xl233 by 0.002 on trained pixels* — a difference read off two pairs whose
  own noise cannot resolve it.

**What this does to "width beyond 144 buys nothing": it SURVIVES, and it is now made on
honest pixels.** E-032's original closure was hedged precisely because it was a null read
off the blended scoreboard, and §3b says *a settled negative is settled only on the
scoreboard that settled it*. The blended scoreboard could be accused of one specific
artefact — that the 24% never-trained block, being a pure extrapolation, might reward the
wider stencil's longer reach and so *manufacture* a width effect (or, symmetrically, mask
one). It does neither. Remove the block entirely and the two rungs converge rather than
separate. The axis stays closed, on a scoreboard that no longer contains the hole. What is
still NOT established is that width is closed for a model trained WITHOUT the hole; that is
#414's question, not this one.

#### 4 · `_holdlon` — the numbers, and why they still may not be quoted

| rung | seed 0 | seed 1 | pair mean | pair \|Δ\| |
|---|---|---|---|---|
| **xl144** — #418 | **0.22058** | **0.20125** | **0.21092** | **0.01933** |
| xl233 — #417 | 0.20592 | 0.19633 | 0.20112 | 0.00958 |

**#418 doubles the largest `_holdlon` pair delta in the record, and the caution in
[E-043d1 §6](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043d1)
gets stronger, not weaker.** 0.01933 is **3.8× the largest blended pair delta ever measured
at this tier** (0.0051, this very pair) and **18× the same pair's own `_trainlon` delta**
(0.00108). Pooled over the two pairs the `_holdlon` seed sd is **0.01079** against
`_trainlon`'s **0.00066** — a **16.4× ratio**, from the same four checkpoints, in the same
two files, at the same horizons. Extrapolating into a training hole is the least
reproducible thing this programme measures, and the split now says so with two pairs
instead of one.

So: **a `_holdlon` number is a mechanism reading, never a level.** Quotable as *"the
stencil head retains roughly a fifth of climatological skill inside the block where the
1-point gate retains essentially none (0.211 vs 0.058)"*; not quotable as *"xl144 scores
0.211 in the block"*, and emphatically not as a rung-to-rung comparison — xl144's 0.21092
against xl233's 0.20112 is a 0.0098 difference against a 0.0108 seed sd, which is nothing
at all.

#### 5 · The stencil advantage decomposes the same way at BOTH rungs — the share replicates

Rerunning #417 §5's computation on the flagship rung, corridor, pair means against the
frozen 1-point gate, five decimals:

| scope of the comparison | xl144 (#418) | xl233 (#417) |
|---|---|---|
| **blended — the published figure** | 0.67813 − 0.58908 = **+0.08904** | **+0.08483** |
| **`_trainlon` — honest pixels** | 0.86754 − 0.80425 = **+0.06329** | **+0.06138** |
| `_holdlon` — the training hole only | 0.21092 − 0.05767 = **+0.15325** | **+0.14346** |
| **share of the published advantage present on trained pixels** | **71.1%** | 72.3% |

Window: +0.07917 blended, **+0.04500** trainlon, +0.16862 holdlon — **56.8% / 43.2%**,
against xl233's 55.0% / 45.0%. Gate scope: +0.08204 / **+0.04175** / +0.20408 — 50.9%.

**Two independent rungs give 71.1% and 72.3% on the corridor, and 56.8% and 55.0% on the
window.** That is the first thing in this arm that replicates across checkpoints rather than
across seeds, and it is the number the paper should carry: **roughly 70% of the corridor
stencil advantage, and only ~56% of the window one, is present on pixels the model trained
on; the rest is what the blend buys by including a hole the 1-point gate cannot see into at
all.** The scope-dependence is itself the mechanism — the window scope contains
proportionally more of the block (25.02% of pixels vs the corridor's 23.93%) and more of the
open ocean where the gate has no neighbours to lean on.

#### 6 · What Table 5 becomes

[E-043d1 §9](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043d1) said
nothing in Table 5 should be revised until #418 landed, because xl144 is the rung the paper
quotes. It has landed. The revision is now available and is **not** done in this entry (the
paper was outside this session's ownership):

- xl144's published corridor **0.68067 / 0.67558** is a LOWER BOUND on the same heads'
  skill over the pixels they trained on, by **+0.18633 / +0.19250** (mean **+0.18942**).
- The retired §0d estimate for xl144 was ≈0.82. **Measured: 0.86754.** The estimate
  understated by **+0.048**, the same direction and nearly the same size as the gate's
  (+0.054) and xl233's (+0.046) misses. The ≈0.75 / ≈0.82 figures stay RETIRED.
- The ranking is unaffected: every arm in `ml/LEADERBOARD.md` is scored over the same
  pixels, so the blend is a common deflation, not a per-arm one.

#### 7 · §3b's band-r row was WRONG and is rewritten in this commit

`amoc_bands` (transport truefit r on rolled section states) is unchanged by the split — the
26.5°N section carries no `_trainlon` block in any head — so #418's values are #356's,
bit-for-bit: xl144 s0 **0.476 / 0.355 / 0.437**, s1 **0.430 / 0.243 / 0.318**, seed spreads
**0.046 / 0.112 / 0.119**.

Those spreads do not fit §3b's *"transport band r spreads 0.05–0.07"* row, and **the row
was never right**: it summarised two replicate groups to one number each, and one of those
two groups was this pair, whose own h4-6 and h7-12 spreads are 0.112 and 0.119 — nearly
twice the top of the quoted regime. So the archive was re-mined rather than patched. Every
xl-tier group with `amoc_bands` in `probes-*.json`, recomputed 2026-08-20:

| group | source | h1-3 | h4-6 | h7-12 |
|---|---|---|---|---|
| E-028 xl55 (**triple**, ranges) | `probes-333` | 0.019 | 0.021 | 0.021 |
| E-031 xl89 | `probes-355` | 0.037 | 0.055 | 0.070 |
| **E-032 xl144** | `probes-356` = `probes-418` | 0.046 | **0.112** | **0.119** |
| E-035 xl233 | `probes-417` | 0.042 | 0.066 | 0.069 |
| E-036 zn × xl144 | `probes-401` | 0.021 | 0.014 | 0.023 |
| E-037 zn × xl233 | `probes-394` | **0.003** | 0.011 | 0.029 |

**Five pairs and one triple — the same replicate groups the corridor-AUC row already
uses.** Two of them (E-036, E-037) had simply never been mined for bands. The range is
**0.003 – 0.119**, and the pooled sd over the fifteen pair-band contrasts is **0.041**
(15 dof) — **20× the corridor's 0.0021 on the identical checkpoints**. §3b's two band-r
rows are replaced by one row carrying all of this, in the same commit as this entry, per
§3b's own rule that the table is extended when a replicate lands.

**#418 itself adds NO new band-r pair.** It reproduces E-032's bands exactly, which is
protocol determinism. What #418 changed is that it forced the row to be checked.

The `_trainlon` and `_holdlon` rows in §3b go from **1 pair to 2** on this entry — that IS
a new replicate group, because #418's checkpoints are different heads from #417's:

| §3b row | pairs | deltas | pooled sd |
|---|---|---|---|
| corridor AUC, `_trainlon` | 2 (xl233 #417, xl144 #418) | 0.00075, 0.00108 | **0.00066** (2 dof) |
| corridor AUC, `_holdlon` | 2 (same) | 0.00958, 0.01933 | **0.01079** (2 dof) |
| corridor AUC, blended, these two pairs | 2 | 0.00200, 0.00509 | 0.00273 (2 dof) |

The ordering is the finding and it is now on two pairs: **`_trainlon` is TIGHTER than the
blended scope, `_holdlon` is 16× looser.** Scoring only trained pixels removes a variance
source; scoring only the hole is almost entirely that variance source.

#### 8 · Cost (§3)

| item | value |
|---|---|
| box | `gpu-box-31479844` (vast 47724559), shared with #417 by design |
| wall clock | 23:39:13Z → 06:10:07Z = **6.51 h** |
| gate head `s1_s0` | **1,993.0 s** |
| xl144 s0 | **10,523.7 s** |
| xl144 s1 | **10,521.2 s** |
| head time, total | **23,037.9 s = 6.40 h** |
| dead dispatches | **none** |
| approx. cost | ~**$1.9** at $0.294/h |

The gate cost 1,993.0 s here against 1,983.0 s in #417 — 0.5% apart on the same box, same
protocol, ten hours apart. Per-step rate over 714 steps per head: **14.7 s/step** for a
stencil-145 head, matching the 14.7–15.5 s/step band #394 / #401 / #413 / #417 established.
Arm D cost **$3.8 and 13.0 h of one box in total** and produced the decomposition of every
published corridor AUC at both top rungs.

#### 9 · What is still open after arm D

Arm D is **COMPLETE**. Both halves landed, the gate reproduced in both, and the paper's
correction is now measured at both rungs rather than estimated at one.

Still open on the wave: **#414 (E-043b: xl144 stage-2 head trained on the ALL-longitude
pool over the existing frozen anchor)** — the decisive arm, and the one #418 is the control
for. #418 says how much the hole deflates a number; only #414 says what the number becomes
when there is no hole. **Note for whoever harvests #414's roll: its blended corridor AUC is
NOT comparable to 0.67813.** A head trained on all longitudes has no `_holdlon` handicap,
so its blended figure should EXCEED this pair's blended figure even if its skill is
identical — the honest comparison is `_trainlon` against `_trainlon` (0.86754) and blended
against blended only as a bound.

---

<a id="e-043b"></a>
### E-043b · #414 — RESULT (training half), completed 2026-08-19; the decisive arm's head exists, and its one-step ratio is NOT yet a result

**E-043b · Trains a FRESH xl144 stage-2 head (1024×16, K 24, sunflower-144 = stencil 145)
for 200,000 steps on the EXISTING frozen `f3_anchor41M` codec, with the stage-2 training
pool opened to ALL 481 longitude columns (`--train-lon-hold none`, recipe
`xl144-nolonhold`) — arm B of the E-043 wave, and the only arm that says what a number
BECOMES when there is no hole · params 40.693M codec (frozen) + **211.353M** head · stage
`stage-2` · data `family3_na025` (C 39, T 516, tensor sha256 `adcbe700fb6e…`) · arch codec
576×10, 8 heads, d_dec 768, d_z 64, patch 3; head 1024×16, K 24, stencil 145, ring
`spiral:111,4444,0.71,0.5` · steps×batch 60,000 × 512 stage-1 (= the checkpoint's own step
count, so NOTHING trains in stage 1) then **200,000 × 256** stage-2, expdecay peak 1e-3,
halflife 40,000, warmup 2,000, cooldown-frac 0 · resume `!run-62,run-63`
(`f3_anchor41M`).**

**Code.** #414 → `head_sha` `8f0e5141`, job `train` on `gpu-box-30257785`, torch
2.13.0+cu126 on an RTX 4090.

[#414 (E-043b xl144 stage-2 head, all-longitude pool) — the CI log](https://github.com/blauewelt/earth/actions/runs/32278072256)

[probes-414.json on ml-metrics](https://github.com/blauewelt/earth/blob/ml-metrics/probes-414.json)

[E-043d2 · #418, this arm's CONTROL pair, decomposed](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043d2)

**Scale (rule 6).** head parameters **211,352,640** (= 211.353M, the E-032 xl144 geometry
exactly, and identical to #346/#347's `scale.params`) · batch **256** · steps **200,000** ·
`data_points` **38,488,680** · 84,405 ocean pixels · 480 train months · stencil **145**.

#### 1 · What this entry does and does NOT settle

**#414 trained a head. It did not roll one.** The corridor AUC that decides this arm comes
from the sroll, which needs the head published first (#420, HEADPUB) and then rolled. This
entry records the training half and the probe ladder that shipped with it, so that the
roll — when it lands — is read against numbers that were written down before it existed.

#### 2 · THREE of the five archived probes carry no information about this run

`probes-414.json` contains `probe_kfold`, `temporal.json`, `probe_sequence`, `dip_check`
and `provenance` — and **`probe_kfold.json`, `probe_sequence.json` and `dip_check.json`
are bit-identical to #346's and #347's**, field for field:

| read-out | #414 | #346 | #347 | what it scores |
|---|---|---|---|---|
| `probe_kfold` rapid r | 0.627 [0.503, 0.735], RMSE 2.17, lp18 0.811 | **identical** | **identical** | the frozen CODEC |
| `probe_sequence` K=1/3/6/12/24 (raw) | 0.426 / 0.448 / 0.522 / 0.599 / 0.625 | **identical** | **identical** | the frozen CODEC |
| `dip_check` r_oof / sign / captured | 0.627 / 68.8% / 47.5% | **identical** | **identical** | the frozen CODEC |

This is the arm working as designed, not a bug: E-043b freezes the anchor and changes only
the stage-2 training pool, so every codec-side read-out MUST be unchanged, and the fact
that all three are unchanged to the last digit is the strongest available confirmation
that **the codec really was untouched** and that the Z cache is byte-identical to
#346/#347's. It also means **only `temporal.json` may be quoted for this run.** Quoting
0.627 as an E-043b result would be quoting the anchor.

#### 3 · The one-step forecast ratio — 0.01392, and why it is NOT a 9× improvement

`temporal.json` `z_t+1`, against the two published xl144 controls:

| run | training pool | `mse_model` | `mse_persistence` | **ratio** |
|---|---|---|---|---|
| **#414** (E-043b, all longitudes) | ALL 481 cols | 0.0436946 | **3.13943290710449** | **0.01392** |
| #346 (E-032 xl144 seed 0) — CONTROL | 401 cols, −45..−25 held out | 0.3879833 | **3.13943290710449** | 0.12358 |
| #347 (E-032 xl144 seed 1) — CONTROL | 401 cols, −45..−25 held out | 0.3797261 | 3.12627625465393 | 0.12147 |

Taken at face value that is **8.88× better than #346 and 8.73× better than #347**. It is
not being taken at face value, and this section exists so that nobody later does.

**(a) The denominator is bit-identical, so the val POPULATION is identical.** #414 and
#346 report `mse_persistence` = 3.13943290710449 to the last bit (#347, a different seed,
differs in the twelfth digit — the seed moves the val sample). Persistence depends only on
the data, so an identical persistence MSE says the two runs scored **the same pixels in
the same months**. That is good news for the comparison's cleanliness and *bad* news for
its interpretation, because it pins down exactly what changed: not the test set, only who
was allowed to train on it.

**(b) That val population contains a block #346 never trained on and #414 did.** The
−45..−25 holdout is **21,120 of 84,405 ocean pixels = 25.02%**. For #346 those pixels are
pure extrapolation; for #414 they are training data. A quarter of the scored domain
switched from "never seen" to "seen", and #418 measured what that switch is worth on the
corridor: in-block h=6 skill ~0.23 against ~0.86 outside, and `_holdlon` pair deltas 16×
looser than `_trainlon`'s off the identical checkpoints.

**(c) The arithmetic says the whole 8.88× is consistent with pure scope.** Split each
run's MSE into trained and held blocks, `m = (1−f)·m_train + f·m_hold` with f = 0.250222.
Ask what must be true for #414 and #346 to have **identical trained-pixel skill** — i.e.
for the entire gap to be scope. Then `m_hold(346) − m_hold(414)` = 0.3442886 / 0.250222 =
**1.3759**, which is satisfiable (it needs `m_hold(346)` ∈ [1.376, 1.551], well inside the
range a 25% block can carry), and it forces the common trained-pixel MSE to be at most
`m_414/(1−f)` = **0.058277**, i.e. a trained-pixel ratio of **0.01856**. So a world in
which the two heads are equally good on trained pixels is fully consistent with these
archives, and in that world **#346's own honest ratio is 0.01856 — within 1.33× of
#414's 0.01392, not 8.88× away.** Nothing in `probes-414.json` can distinguish that world
from a real improvement, because `z_t+1` is not split by longitude scope.

**HONEST STATEMENT: the one-step ratio is NOT comparable until the sroll lands.** It is a
scope artefact of unknown size, in exactly the direction and for exactly the reason #418's
closing note warned about for the corridor AUC. It must not be reported as a
forecasting-improvement claim, in a session summary or anywhere else.

**(d) The pixel-space number moves far less, which is itself a clue.** `chan_t+1` (the
same forecast read out in channel space, same bit-identical persistence 1.15408134460449)
goes 0.19979 → 0.05674, a factor of **3.52** against `z_t+1`'s 8.88. Two read-outs of one
forecast disagreeing by 2.5× on the size of the effect is what a scope artefact looks like
when the two spaces weight the held block differently; it is not what a uniform skill gain
looks like.

#### 4 · `rapid_probe_kfold` 0.389 — POOLED, and below both controls

| run | `rapid_probe_kfold` r | 95% CI | RMSE Sv |
|---|---|---|---|
| **#414** | **0.389** | [0.255, 0.509] | 2.60 |
| #346 (CONTROL) | 0.437 | [0.297, 0.556] | — |
| #347 (CONTROL) | 0.429 | [0.294, 0.544] | — |

#414 sits **0.044 below the control pair's mean (0.433)**, whose own seed range is 0.008.

Two labels, both required by standing rules, and neither optional:

- **POOLED, therefore distrusted.** §3: `rapid_probe_kfold` is section-pooled —
  `temporal.py:2018` does `hid[:, -1].mean(0)` — and geostrophic transport is the
  east-minus-west contrast ACROSS 26.5°N, which a mean annihilates. This is the
  comparable-to-history number, **never the verdict**.
- **NOT a distinguishable difference.** §3b has no xl-tier row for the RAPID head k-fold;
  the nearest measured spread is the 1.8–10.7M · 60k pooled sd of **0.095** (10 dof), and
  the 1.8M · 6k triples span **0.245**. A 0.044 gap is well inside both. It is recorded,
  it is not a finding, and it is certainly not evidence that opening the pool hurt the
  transport read-out.

`rapid_probe` (the 36-month split, n_test 36) reads raw 0.458 / deseasonalised 0.372
against #346's 0.460 / 0.328 and #347's 0.510 / 0.316 — labelled as the non-k-fold number
§3 requires it to be labelled as, and used for nothing.

#### 5 · GAP — there is no `probe_head.json` for this run, and the cause is in the inputs

**`probes-414.json` has no `probe_head.json`.** The cause is not a crash and not a missing
file: `provenance.json` records **`head_probe: "false"`** in the dispatch inputs.

**How it got there.** §1 requires a replication to *copy the full INPUTS_JSON block out of
the log of the run it is replicating* — the rule that exists because run #395 died in 90
seconds with sixty `size mismatch` lines from hand-assembling a dispatch. #414's inputs
were copied verbatim from #346, and **#346 predates the head probe**: its own provenance
also carries `head_probe: "false"`. The rule worked exactly as written and carried a
stale-but-valid field forward with everything else.

**Is it a rule violation?** No — §3's "pass `head_probe: true` on every eval dispatch" is
scoped to *the new cadences* (pentad/daily), and #414 is monthly `family3_na025`. It is
still a **gap worth closing**, because §3's KNOWN GAP paragraph applies here at full
force: this run's only stage-2 transport read-out (§4 above) is the section-pooled one,
which is the read-out §3 says not to trust as a verdict. So the decisive arm currently has
no unpooled transport number at all.

**The cheap closure, priced and NOT dispatched.** An **eval-only** re-dispatch against the
head #420 publishes, with `head_probe: "true"`, `temporal_steps: 0` and `max_minutes: 0`
— nothing trains, the Z cache is warm and byte-identical, and the pattern is #409's. It
costs one short job, not a retrain. **Deliberately not dispatched in this session**: the
session's dispatch budget went to the sroll, which answers the arm's actual question,
and stacking an eval behind it on a box mid-roll would only delay the number that matters.
Recorded here so the next session can spend it in one step.

#### 6 · What decides this arm, and the reading discipline it must be read under

The verdict is the sroll of `e043b-xl144-nolonhold-s0` against the frozen `e017_u1_s0`
gate. **Its BLENDED corridor AUC will exceed the control pair's blended 0.68067/0.67558
even at identical skill**, because this head has no `_holdlon` handicap to drag its
average down — the control pair's blended figure is deflated by a 24% block it never
trained on, and #414's is not. Reading that excess as an improvement would be the same
mistake as §3(c) above, made twice in one wave.

The comparisons that mean something:

1. **`_trainlon` against `_trainlon`** — #414's trained-pixel corridor AUC against the
   control pair's **0.86700 / 0.86808, mean 0.86754** (#418). Both sides honest pixels,
   both sides fully trained. This is the arm's headline.
2. **Gate-anchored** — `e017_u1_s0` must reproduce 0.643 within `GATE_TOL` 0.0101 in
   #414's roll as it did in #417's and #418's, or the roll is void and no number leaves it.
3. **Blended against blended only as an upper bound**, explicitly labelled as one.

And the falsifier, restated from the dispatch and unchanged by anything above: if the
all-longitude head does not differ from the control pair beyond the xl tier's band
(§3b pooled sd **0.0021** on 7 dof, 95% upper bound **0.0037**), the longitude holdout was
costing stage 2 nothing, and the 48%/59% hole-patching decomposition of the stencil head's
advantage needs re-reading. **ONE SEED** (Chris, 2026-08-19); an effect landing near the
±0.025 corridor-AUC bar buys its second seed then, not now.

---

<a id="e-043b-roll"></a>
### E-043b · #420 (HEADPUB) and #422 (the roll) — RESULT, completed 2026-08-20 13:10:03Z. Sections 1–5 are the pre-registration written at dispatch; **the result is §6, and §7 is why its headline number is not yet readable as a level**. #421 was the same roll, VOID, and is recorded in §2b

Sections 1–5 were written **at dispatch**, hypothesis and falsifier first, so the log could
not be rewritten to fit the answer (§1). They are left exactly as they were written.
**#422 landed at 13:10:03Z; its result is §6 and the reading discipline it forced is §7.**

**E-043b-roll · ROLLS the freshly published e043b xl144 stage-2 head
(`head-weights-e043b-xl144-nolonhold-s0`, trained by #414 for the full 200,000 steps on the
EXISTING frozen `f3_anchor41M` codec with the stage-2 pool opened to ALL 481 longitude
columns) twelve months forward beside the frozen `e017_u1_s0` validation gate, reading the
`*_trainlon` / `*_holdlon` split in every scope — the VERDICT of arm B · params 40.693M
codec (frozen) + 211.353M head · stage `sroll` (eval-only; NOTHING trains, the step field
is the codec checkpoint's own count) · data `family3_na025` (C 39, T 516, sha256
`adcbe700fb6e…`) · arch codec 576×10, 8 heads, d_dec 768, d_z 64, patch 3; head 1024×16,
K 24, stencil 145, ring `spiral:111,4444,0.71,0.5` · steps×batch 60,000 × 512 (=
f3_anchor41M's recorded step count, ZERO training steps) · resume `!run-62,run-63`
(f3_anchor41M, frozen).**

[#422 (E-043b roll, gate + the all-longitude xl144 head) — the CI log](https://github.com/blauewelt/earth/actions/runs/32354718326)

[#421 (E-043b roll, first attempt — VOID, rolled the gate alone) — the CI log](https://github.com/blauewelt/earth/actions/runs/32348876544)

[#420 (HEADPUB `e043b-xl144-nolonhold-s0`) — the CI log](https://github.com/blauewelt/earth/actions/runs/32345210849)

[E-043b · #414, the head this rolls](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043b)

[E-043d2 · #418, the control pair's `_trainlon` decomposition](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043d2)

#### 1 · #420 — the head was PUBLISHED, and the publication was verified before it was rolled

#420 completed **2026-08-20 08:08:14Z** on `gpu-box-30257785`, window
`headpub:e043b-xl144-nolonhold-s0@temporal`, asset
**`head-weights-e043b-xl144-nolonhold-s0.pt`**, 845,487,479 bytes, upload **HTTP 201**,
asset `state` polled to **`uploaded`**.

§0.2 says verify the ARTEFACT, and the specific failure to rule out is #378/#381/#382 —
three heads published from a **stale `orphan-temporal-latest.pt` leftover** and given the
names of arms that had not produced them. Five independent checks, every one of them a
measurement rather than an inference:

| check | what would have failed | what was read |
|---|---|---|
| source pinned | `@temporal` vs orphan-first default | log: `source: /opt/earth-cache/ckpt/temporal.pt (2.4G)` — **the orphan path was never consulted** |
| the box is #414's box | a head from some other run | #414 ran on `gpu-box-30257785`; #420 ran on `gpu-box-30257785` |
| nothing intervened | a later run overwriting `temporal.pt` | #414 completed **07:41:51Z**, #420 started **07:42:06Z** — a **15-second** gap, and no other run touched that box in between |
| the file's own fields | wrong arch / wrong seed / a milestone asset | the box's own `torch.load` printed **`step=200000 d_model=1024 layers=16 stencil=145 seed=0 znoise=0.0`**, **`params=211.4M`** — matching #414's `temporal.json` `scale.params` **211,352,640** exactly, and `step=200000` rules out the 600 / 60,000 / 120,000 milestone assets |
| no confusable leftover exists | those six fields are IDENTICAL for #346 | #346 ran on `gpu-box-47566393` and #347 on `gpu-box-46045353` — **neither control head has ever run on `gpu-box-30257785`**, so no file with that field signature could be sitting there |
| **the file itself, opened here** | any of the above being a misreading of a log | the 845,487,479-byte asset was downloaded at 09:35Z and `torch.load`-ed locally: `d_model` **1024**, `layers` **16**, **`K` 24**, `stencil` **145**, `seed` **0**, `ring_km` `spiral:111,4444,0.71,0.5`, `pos.weight` rows **24**, `params` **211,352,640**, and `steps` 200,000 / `batch` 256 / `lr` 1e-3 `expdecay` halflife 40,000 warmup 2,000 / `milestone_steps` `600,60000,120000` — #414's dispatch, field for field |
| **`train_lon_hold` = `'none'`** | THE decisive field, and the only one that separates #414's head from #346's | read out of the file's own `args`: **`'none'`**. #346 trained with the −45..−25 block held out; this head did not. Nothing else in the checkpoint distinguishes them |

The last two rows are the ones that matter, and they are what the publish script's printed
line CANNOT settle by itself: **#346 is also 1024×16, stencil 145, seed 0, znoise 0,
200,000 steps, 211.4M.** The six printed fields do not distinguish #414's head from #346's.
Two things do — the BOX (no #346-shaped file has ever existed on `gpu-box-30257785`) and
**`train_lon_hold`**, which is `'none'` here and is the entire subject of this experiment.
Both were checked. Note the ordering forced by §0.2: the box argument was available
immediately and the file could not be opened until 09:35Z, because the CDN was serving a
cached 404 (§2b) — so the run was verified by provenance first and by the artefact second,
and the artefact confirmed the provenance.

**The asset name is `head-weights-<tag>.pt`, not `<tag>.pt`.** `scripts/sroll_run.sh` tries
`<tag>__temporal.pt` and then `<tag>` verbatim, so the roll's window token must be the FULL
asset stem — `sroll:e017_u1_s0,head-weights-e043b-xl144-nolonhold-s0`, exactly as #413,
#394 and #401 wrote it. A window naming the bare arm (`…,e043b-xl144-nolonhold-s0`) 404s on
both attempts and the script emits only `::warning::head … not on the release — skipped`;
with the gate still fetched, `[ -n "$HPATHS" ]` passes and the run goes **GREEN having
rolled nothing but the gate**. Checked at dispatch, where the inputs are all it costs (§0.3).

#### 2 · A CDN hazard this session created, recorded because the next one can trip it

The asset record appears in the GitHub API with its final size **while `state` is still
`starter`** — the blob is not yet fetchable. A download attempted in that window returns
Azure `BlobNotFound`, and **GitHub's Fastly edge caches that 404 on the asset path,
ignoring the signed query string**. Measured here: the premature request at 07:58:19Z was
still being served from `cache-iad-kcgs7200068-IAD` with `X-Cache: HIT` and a monotonically
rising `Age` past 1,536 s, through fresh signed URLs, `Cache-Control: no-cache`, byte-range
requests and an added cache-buster parameter. **Poll `state` until `uploaded` before the
first GET.** The full measurement is in §2b.

> **CORRECTED, 09:40Z — the paragraph that stood here was WRONG, and #421 falsified it
> within twenty minutes.** It read: *"The roll itself is not believed to be affected — box
> `47724559` is in Quebec, CA and resolves a different POP, which was never poisoned
> because only IAD ever saw the bad request."* That was an inference from a geolocation
> field, not a measurement, and it is exactly the class of reasoning §0.1 forbids — the
> risk was named, judged low on a guess about CDN topology, and dispatched against. **#421
> then hit a 404 on that head and rolled the gate alone** (§2b). The box had made its own
> premature request and poisoned its own POP; "only IAD ever saw the bad request" was
> false the moment the job started. The correct move, taken for #422, was to wait out the
> hour and confirm `HTTP 200` with the exact bare-`curl` call the box makes before
> dispatching — a check that costs one command. It is left visible rather than deleted
> because the shape of the error is the lesson: a precondition was downgraded to a
> probability estimate, and the estimate was wrong.

#### 2b · #421 — the SAME roll, VOID: it rolled the gate alone and would have gone green

**#421** (E-043b roll, first attempt) was dispatched **08:27:11Z** at `head_sha` `c3bdc95`
with an identical window and was **CANCELLED at 08:44Z**, 17 minutes in, having produced
nothing. It is recorded in full because the failure is invisible by construction and the
next session will meet it again.

**What it did.** The window named two heads. `scripts/sroll_run.sh` fetched the gate,
**404'd on `head-weights-e043b-xl144-nolonhold-s0`**, emitted
`::warning::head … not on the release — skipped`, and carried on: `[ -n "$HPATHS" ]`
passes as long as ONE head arrived, and the gate always arrives. The roll started with
`n_heads = 1`, and that one head was the gate.

**Why nothing would have caught it.** Left alone it would have run ~30 minutes more,
**passed its gate**, written a well-formed `rollout_spatial.json`, satisfied every one of
the script's closing assertions — they check the heads that ARE present, and the gate was
present and perfect — and archived a `probes-421.json` indistinguishable from a good run.
It was caught by reading the live metrics: **`"heads": 1`**, and the head's label
**`s1_s0`** — stencil 1, seed 0, which is the 1-point GATE, not the 211M stencil-145 head
the run existed for. A run number, a green tick and a passing gate said nothing.

**Why the fetch 404'd, and this is the part worth carrying forward.** A GitHub release
asset **appears in the API with its final size while its `state` is still `starter`** —
the blob is not yet fetchable, and a GET in that window returns Azure `BlobNotFound`.
**GitHub's Fastly edge then caches that 404 on the asset PATH, ignoring the signed query
string.** Measured here rather than assumed:

| observation | value |
|---|---|
| asset created / `state: starter` | 07:55:21Z, 845,487,479 bytes |
| upload completed (`HTTP 201`, step 21 green) | 08:07:36Z |
| `state` polled to `uploaded` | 08:07:59Z |
| a premature GET | **07:58:19Z** — the poisoning request, made by this session |
| what it served afterwards | `404 BlobNotFound`, `X-Cache: HIT`, `X-Ms-Request-Id` frozen at the 07:58:19Z request, from `cache-iad-…-IAD` |
| defeated by | fresh signed URLs · `Cache-Control: no-cache` · `Pragma: no-cache` · byte-range requests · an added cache-buster parameter — **all of them, because the cache key is the path** |
| expired at | `Age` **3,691 s** — Fastly's default hour. The very next request was `200 OK`, `X-Cache: MISS` |

The box made its OWN premature request at ~08:30 and poisoned whichever POP serves it, so
#422 was held until **09:35Z** — past both hours — and dispatched only after the asset was
confirmed `HTTP 200` with the exact bare-`curl` call `sroll_run.sh` makes.

**Rule: poll the asset's `state` until `uploaded` BEFORE the first GET, from anywhere.**
The size in the API is not evidence the bytes are there, and the POP that serves the box is
poisoned by whoever asked first — including a session merely trying to verify the artefact.

**The fix, shipped in the same session (main `32e4e06`).** A named head that does not
arrive is now a **REFUSAL**, not a warning: `sroll_run.sh` collects the missing tags, exits
1 naming each one and both asset-name conventions it tried, and prints
`heads: N named, M fetched` beside the byte count of each head it did get — all before the
roll starts, where the inputs are the only thing it has cost (§0.2, §1). The old behaviour
existed so one bad tag could not lose a multi-head roll; that trade is wrong at this price,
and a two-minute refusal is cheaper than a void six-hour roll discovered at harvest.
`tests/test_sroll_wiring.py` **case 3b** is the regression itself — a `curl` stub that
succeeds for one tag and 404s the other must take the run down and must NAME the missing
tag. All 8 checks hold.

**Cost of the void run:** 17 minutes of one box, ~**$0.08**, plus ~1.1 h of idle burn on
`gpu-box-31479844` (~$0.32) across the cancel and the wait for the cache hour — during
part of which the hourly health check correctly stopped the box, which is why the figure is
not higher. GitHub had already discarded #421's logs by the time they were fetched (the
archive returns a 22-byte empty zip), so the evidence above is the live-metrics records and
a local reproduction of the box's exact `curl`, not the job log.

#### 3 · READING DISCIPLINE — the blended number is a trap, and it is set before the answer exists

**This head trained on ALL longitudes and therefore carries no `_holdlon` handicap.** Its
**blended** corridor AUC will EXCEED the published E-032 xl144 pair's blended
**0.68067 / 0.67558** *even at identical skill*, because the controls' blended figures are
deflated by a 24% block they never trained on and this one's is not. **A blended-vs-blended
excess is not evidence of improvement and must not be reported as one.**

The comparisons that mean something, in order:

1. **HEADLINE — `_trainlon` against `_trainlon`.** This head's trained-pixel corridor AUC
   against the control pair's **0.86700 / 0.86808, mean 0.86754** (#418). Both sides honest
   pixels, both sides fully trained.
2. **GATE-ANCHORED.** `e017_u1_s0` must reproduce **0.643** within `GATE_TOL` **0.0101**, as
   it did in #417 and #418, or the roll is **VOID** and no number leaves it.
3. **Blended against blended — admissible only as an explicitly labelled upper bound.**

§3b's `_holdlon` row applies with full force to whatever this run's `_holdlon` figure is:
2 pairs, pooled sd **0.01079**, 16.4× the `_trainlon` sd off identical checkpoints, and
**never quotable as a level**. This run adds no pair to that row — it is one head.

The same trap has already caught this arm once, in the one-step ratio: #414's `z_t+1`
0.01392 against #346's 0.12358 looks like 8.9×, and the arithmetic shows the whole of it is
consistent with pure scope (see [E-043b · #414
§3](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043b)). Twice in one
wave would be a pattern, not an accident.

#### 4 · Hypothesis and falsifier

**HYPOTHESIS.** Opening the stage-2 training pool to all longitudes leaves **trained-pixel
skill essentially unchanged** — the head already saw 76% of the corridor and the anchor
codec is untouched, so there is no new information about those pixels — while lifting
`_holdlon` sharply, because the −45..−25 block stops being extrapolation.

**FALSIFIER, stated before the number exists.** If `_trainlon` differs from **0.86754** by
more than the xl tier's band (§3b pooled sd **0.0021** on 7 dof, 95% upper bound
**0.0037**), then the longitude holdout **was** costing stage 2 real skill on pixels it had
already trained on — a genuinely surprising result, since it would mean the held block was
carrying information the trained pixels needed — and the 48% / 59% hole-patching
decomposition of the stencil head's advantage needs re-reading.

**ONE SEED**, per Chris 2026-08-19. §3b governs the reading: an effect landing near the
±0.025 corridor-AUC bar buys its second seed then, not now. **NOT A REPLICATE** — the
gate's reproduction measures the PROTOCOL and may not be entered in §3b's spread column.

#### 5 · Cost and the standing gap

Budget: gate + one head. #418 spent **1,993.0 s** on the gate and **~10,522 s** per xl144
head, so this run is **~3.5 h** of head time against a `job_timeout` of **700 minutes**.
The box (`gpu-box-31479844`, vast **47724559**, Quebec CA, $0.294/h) keeps #418's warm Z
cache and tensor, so there is no embed pass.

Still open, and deliberately not spent here: **#414 has no `probe_head.json`** (its dispatch
carried `head_probe: "false"`, copied from #346 which predates the probe). The cheap closure
is an eval-only re-dispatch against this same published asset with `head_probe: "true"` —
see [E-043b · #414 §5](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043b).

---

#### 6 · RESULT — #422 completed 2026-08-20 13:10:03Z. The gate passed, the head rolls `_trainlon` at **0.93933**, and it does so with a lead-time profile unlike anything else in the archive

**Code.** #422 → `head_sha` `32e4e06`, job `sroll:` on `gpu-box-31479844` (vast 47724559),
torch 2.13.0+cu126 on an RTX 4090. Gate wall **1,988.5 s**, head wall **10,497.9 s**.

[probes-422.json on ml-metrics](https://github.com/blauewelt/earth/blob/ml-metrics/probes-422.json)

**Gate — PASSED.** `e017_u1_s0` returned `horizon_auc` **0.643** (recomputed from its own
twelve `msss_clim` values: **0.64283**) and bands **0.470 / 0.375 / 0.492**, `pass: true`,
`fails: []`, against `GATE_REF` 0.643 / `tol` 0.0101. Its corridor scopes reproduce #417's
and #418's to the last digit — **0.58908 blended / 0.80425 `_trainlon` / 0.05767
`_holdlon`** — so this is the **twentieth** reproduction of the number (#228 … #418 → #422).
Per §3b that measures the PROTOCOL and is **not** a replicate; it does not enter the spread
column. What it buys is that the numbers below are readable at all.

##### 6.1 · The numbers, recomputed to five decimals from the archived per-horizon arrays

Every figure below is the mean of the twelve archived `msss_clim` values in that scope
block, not the file's rounded `horizon_auc` field (§3b's convention).

| scope | `e043b-xl144-nolonhold-s0` (#422) | n_px |
|---|---|---|
| corridor, **blended** | **0.93733** | 29,627 |
| corridor, **`_trainlon`** | **0.93933** | 22,538 |
| corridor, **`_holdlon`** | **0.93283** | 7,089 |
| window, blended / `_trainlon` / `_holdlon` | 0.93550 / 0.93625 / 0.93375 | 84,405 / 63,285 / 21,120 |
| gate scope, blended / `_trainlon` / `_holdlon` | 0.93242 / 0.93350 / 0.92900 | 864 / 635 / 229 |

**Precision first, because "the hole is gone" is easy to over-read: only STAGE 2's hole is
gone.** The codec under this head is the unchanged frozen `f3_anchor41M`, trained by #62/#63
**with** the −45..−25 block held out. So `_holdlon` here labels pixels the ENCODER still
never saw and the temporal head did — which is exactly arm B's design, and it is why the
comparison is informative at all. (Arm A, **#416**, is the run that removes the codec's hole,
and it has no head.) The scope split itself is the same 80 columns / 7,089 corridor pixels
as #417 and #418, from the same `holdout_lon` block and the same `corridor_def`.

**Given that, skill is uniform across the basin.** The `_trainlon` − `_holdlon` gap is
**0.0065** on the corridor. Both `_holdlon` figures below are quoted as a MECHANISM CONTRAST
and never as levels — §3b's `_holdlon` row (2 pairs, pooled sd 0.01079, 16.4× the `_trainlon`
sd) forbids the second reading and nothing here asks for it. In the same run the gate head's gap is **0.746** (0.80425 → 0.05767); in #418 the
control pair's gaps are **0.646** and **0.667**. The hole that
[§0d · the skill map's central band](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#holdout-lon-band-2026-08-19)
identified as the dominant feature of every published skill map **is not present in this
head's map**. That is arm B's hypothesis confirmed on its second clause, and it is the one
claim in this section that nothing below qualifies.

##### 6.2 · The honest comparison — `_trainlon` against `_trainlon` (§3 pre-registration, comparison 1)

| | corridor `_trainlon` |
|---|---|
| **#422** — all-longitude pool, seed 0 | **0.93933** |
| #418 — E-032 xl144 seed 0 (CONTROL) | 0.86700 |
| #418 — E-032 xl144 seed 1 (CONTROL) | 0.86808 |
| **control pair mean** | **0.86754** |
| **Δ** | **+0.07179** |

Against the bars §3b names: **34.2×** the xl tier's blended pooled sd (0.0021, 7 dof),
**108×** the `_trainlon` pooled sd (0.00066, 2 dof, the tightest scope in the record), and
**2.87×** the ±0.025 one-seed bar. **The pre-registered falsifier fired.** §4 said that if
`_trainlon` differed from 0.86754 by more than 0.0037 then the longitude holdout *was*
costing stage 2 real skill on pixels it had already trained on, and that this would be "a
genuinely surprising result". It differs by 19× that threshold.

**Blended against blended, labelled as the upper bound §3 said it could only ever be:**
0.93733 against the control pair's 0.68067 / 0.67558 (mean 0.678125) — **+0.25921**, of
which the great majority is the hole closing rather than skill moving, exactly as the
pre-registration predicted. It also exceeds the best blended number the programme has ever
published, E-037's xl233 + znoise 0.7 seed 0 at **0.725** (pair mean 0.7240), by **+0.212**
— but that comparison is blended-against-blended too, and the znoise pair was rolled with
the hole in it, so it inherits the same deflation and the same label. **Neither blended
figure is admissible as evidence of improvement.**

##### 6.3 · `horizon_auc_daymatched` — the field's first appearance on real data, and at monthly it is an identity

`e9f3d8d` added `horizon_auc_daymatched` beside `horizon_auc` in every scope block, and
#422 is the first REAL run to carry it (#417 and #418 predate it — the field is absent from
their bundles). Checked on all **eighteen** scope blocks in this run, across both heads:
**`horizon_auc_daymatched` equals `horizon_auc` in every one.** That is the correct
behaviour and it is an identity by construction — at monthly the twelve day-matched leads
ARE h = 1 … 12, so the two means are over the same set. The field's value is entirely at
pentad (spec §7b(g)); what #422 establishes is that adding it moved no monthly number.

##### 6.4 · Transport: `amoc_bands`, the 20-year hindcast, the future roll — **and none of them moved**

| read-out | #422 (all-lon head) | #418 xl144 s0 | #418 xl144 s1 |
|---|---|---|---|
| `amoc_bands` h1-3 r (n 99) | 0.483 | 0.476 | 0.430 |
| `amoc_bands` h4-6 r (n 72) | 0.380 | 0.355 | 0.243 |
| `amoc_bands` h7-12 r (n 63) | 0.498 | 0.437 | 0.318 |
| `long` r_trained (n 195) | 0.790 | 0.780 | 0.787 |
| `long` r_heldout (n 36) | 0.469 | 0.417 | 0.350 |
| `long` r_lp18 | 0.876 | 0.852 | 0.822 |
| `long` amp_lp18 | 0.741 | 0.739 | 0.755 |

`long.context_end` `2004-12`, 240 rolled months; `future.context_end` `2024-12`, 240 rolled
months, `sv_des` present and finite throughout, no correlation block (the future roll has no
truth to score against, by construction — it carries `context_end`, `roll_ym` and `sv_des`
only, as it does in #417 and #418).

**Every one of these is inside the control pair's own spread.** §3b's band-r row prices that
spread at pooled sd **0.041** over 15 pair-band contrasts, and the largest deviation in the
table (h7-12, +0.061 against s0) is 1.5 sd with the s1 twin 0.119 below s0 on the same band.
`long.r_heldout` at n = 36 spans 0.350–0.417 across the control pair alone. **A head that
had genuinely gained 0.072 of corridor AUC over its control has produced no measurable gain
in any AMOC-derived quantity computed from the same rolled states.** That is not a
qualification; it is §7.

#### 7 · WHY THIS NUMBER IS NOT YET A RESULT — the lead-time profile, stated before anybody quotes 0.939

**#422 is green, its gate passed, its artefact is well-formed, and its headline number
should not leave this section unqualified.** §0.2 — *a step that reports success is not
evidence it did anything* — and §0b's rule that diagnosis belongs where the surprise was
noticed. What follows is measurement, not speculation; the mechanism is NOT diagnosed.

**(a) The skill does not decay with lead time. Nothing else in the archive does that.**
Corridor `msss_clim`, h = 1 … 12:

| head | h1 | h2 | h3 | h4 | h5 | h6 | h7 | h8 | h9 | h10 | h11 | h12 | h1 − h12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **#422 all-lon** | 0.940 | 0.939 | 0.939 | 0.939 | 0.939 | 0.938 | 0.938 | 0.937 | 0.935 | 0.935 | 0.933 | 0.936 | **0.004** |
| #418 xl144 s0 | 0.774 | 0.730 | 0.706 | 0.692 | 0.682 | 0.669 | 0.663 | 0.658 | 0.658 | 0.648 | 0.638 | 0.650 | 0.124 |
| gate `e017_u1_s0` (same run) | 0.713 | 0.663 | 0.634 | 0.610 | 0.597 | 0.584 | 0.575 | 0.559 | 0.553 | 0.538 | 0.522 | 0.521 | 0.192 |

`acc` is flat at **0.966–0.970** and `amp_ratio` flat at **0.933–0.954**, where the control
decays 0.883 → 0.751. An anomaly correlation of 0.97 over 29,627 North Atlantic pixels at a
twelve-month lead is not a forecast skill this programme has any basis to expect.

**(b) The h = 1 value IS coherent with what #414 trained.** `1 − chan_t+1.mse_model /
mse_persistence`-scale arithmetic reproduces it on both sides: #414's `chan_t+1` mse_model
0.06548 against a ~1.0 climatological variance predicts h1 ≈ 0.935 and the roll measured
**0.940**; #346's 0.23057 predicts ≈ 0.769 and #418 measured **0.774**. So the roll is not
disagreeing with the training run. **The anomaly is entirely in the eleven steps after the
first: the control's error grows 1.55× over the roll and this head's grows 1.07×.**

**(c) The gain is uniform across all 39 channels, including ones the model cannot forecast.**
From `audit.per_channel_msss_clim_corridor`, h1 / h6 / h12: `tau_x` (NCEP wind stress, an
external forcing field) **0.906 / 0.902 / 0.935** against the control's 0.753 / 0.774 /
0.810; `tau_y` 0.909 / 0.894 / 0.909; `ssh` 0.967 / 0.967 / 0.966. Every channel rises and
every channel flattens. A capability gain from 33% more training pixels does not have that
shape.

**(d) And the transport read-outs show none of it** — §6.4 above, plus **#414's own
`rapid_probe_kfold` 0.389 [0.255, 0.509] sits BELOW both controls' 0.437 / 0.429**. Pooled
and therefore distrusted (§3), and 0.044 is well inside the read-out's noise, so this is not
evidence of harm — but it is emphatically not corroboration of a +0.072 skill gain either.

**What this does and does not license.**

- ✅ **Arm B's hypothesis is confirmed on the hole:** the `_trainlon` − `_holdlon` gap
  collapses from ~0.65 to **0.0065**. That is a statement about WHERE skill sits, it is
  supported by the pixel inventory, and it is robust to everything in this section.
- ❌ **`0.939` may not be written as a level, quoted as "the programme's best rolled
  number", or put in the paper** — not from one seed, and not while (a)–(d) stand. §3b
  already forbade the first two independently: *"any number that will be quoted as a
  headline in the paper, whatever its size"* takes two seeds, and this head is n = 1 on a
  training pool no other head shares.
- ❌ **The falsifier firing is not yet "the longitude holdout was costing stage 2 real
  skill".** It is "`_trainlon` moved 19× further than the band allows, in a run whose
  lead-time profile is unprecedented and whose transport read-outs did not move." Those are
  two different findings and only the second one is measured.

**The three cheapest next steps, priced, none dispatched here.**

1. **A second `sroll:` of the identical published asset** (~3.5 h, ~$1.0 on a warm box).
   Protocol determinism is established for the gate over eighteen runs but has never been
   checked for THIS head; a bit-identical repeat rules out a one-off roll fault and costs
   less than an hour of argument.
2. **The eval-only ladder with `head_probe: "true"`** over
   `head-weights-e043b-xl144-nolonhold-s0` — already priced in
   [E-043b · #414 §5](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043b),
   ~2 h, ~$0.6, and it is the only way to get an UNPOOLED transport number for the decisive
   arm. If the corridor gain is real, the unpooled head probe is where it should show.
3. **The seed-1 arm** — `xl144-nolonhold` at `seed:1`, ~15.6 h and ~$4.6 plus its roll.
   §3b requires it before any level is quoted; it is the expensive one and it should be
   bought only after 1 and 2 have said the number survives.

#### 8 · Cost

| item | value |
|---|---|
| box | `gpu-box-31479844` (vast 47724559, Quebec CA, $0.294/h) — #417/#418's box, Z and tensor warm, no embed pass |
| wall | gate 1,988.5 s + head 10,497.9 s = **3.47 h** |
| money | **≈ $1.02** |
| dead dispatches charged to this arm | **#421**, ~$0.08 + ~$0.32 idle — see §2b |
| head roll rate | 10,497.9 s / 714 steps = **14.70 s/step**, the low end of the #394 / #401 / #413 / #417 band (14.7–15.5) and the fourth confirmation of the rate E-044's pentad arithmetic is built on |

<a id="e-043b-roll-addendum"></a>
#### 9 · ADDENDUM, 2026-08-20 16:15–17:00Z — the ROLL CODE is not the mechanism, and the anomaly is older than the roll

§7 named the lead-time profile and said the mechanism was not diagnosed. The leading
hypothesis afterwards was a wiring slip in the roll rewrite: **#422 is the first code any
RING-STENCIL head has ever rolled** (#418 rolled the same-architecture E-032 pair at
`8f0e514`, before `e9f3d8d`/`ef62fbf` landed, and decayed normally), so if the skill loop's
neighbour gather read OBSERVED Z at rolled timesteps for the ring path, the centre pixel
would roll while 144 neighbours carried ground truth — one-step skill at every horizon,
flat, wind included, and the stencil-1 gate untouched. That story fits every symptom.

**It is false, and three independent measurements say so.** None of them is a reading of
intent; §0.1.

**(i) The roll machinery is BYTE-IDENTICAL across the rewrite.** `diff` of `8f0e514` against
`32e4e06` over `ml/rollout_spatial.py` (833 insertions, all of it `TimeAxis`, bands and
starts) leaves four regions untouched, and they are exactly the regions the hypothesis is
about: **`roll_step`** — old `306–355`, new `714–763`, the ONLY place a neighbour gather
happens (`zj = Zwin[nbr.clamp(min=0)]`, new `752`) — is identical line for line;
**`zwin_from_true`** and **`geometry()`** (old `830–859`, new `1434–1460`) are identical;
the head-load block (old `871–898`, new `1567–1594`) is identical. There is exactly ONE Z
buffer in each loop, `Zwin`, and both the skill loop (new `1662`) and `long_roll` (new
`1813`) advance it the same way, `torch.cat([Zwin[:, 1:], zhat[:, None]], 1)`. The gather
reads that buffer and nothing else. What DID change in the skill loop is the start
enumeration (`ax.starts_for_year` for `for s_off in range(12)`), the season token
(`ax.moy_of_row(t_tgt)` for `(cur[-1] + 1) % 12`) and the RAPID key — none of which touches
where a neighbour's Z comes from, and all of which are identities at monthly.

**(ii) `tests/test_roll_ring_identity.py` (NEW) — the ring path reproduces `BASE_SHA` bit for
bit, and the test can detect the bug that was hypothesised.** The gap that made this
expensive is real and is now closed: `tests/test_roll_monthly_identity.py` pins the monthly
artefact on a fixture carrying **stencil 1 and stencil 9 with `ring_km` UNSET**, so
`_ring_on` is false for both and the `spiral:` branch of `build_stencil` — the geometry
every published stage-2 head since E-032 actually uses — had never once been compared
across a change to this file. The new test adds a third head to the same toy whose only
difference from the fixture's stencil-9 head is `ring_km` (`spiral:200,900,0.71,0.5`; small
radii because the toy ocean is 8×10 cells), rolls it under both versions, and asserts three
things: the ring head was SCORED (its label carries the spiral spec — the #421 shape of
failure, a head silently absent, would otherwise leave both sides equal and the test green
over nothing); its record DIFFERS from the same-slot-count fixed-table head's, so `ring_km`
reached the geometry and the test is not comparing one geometry with itself; and the two
payloads are **BYTE-IDENTICAL** after the same two documented exclusions
(27 `horizon_auc_daymatched` keys, each asserted equal to its scope's `horizon_auc` before
being dropped, plus the `wall_s` clock readings). All three hold. **And the test is not
vacuous:** against a copy of `ml/` in which the skill loop was edited to do exactly what the
hypothesis describes — `Zwin = zwin_from_true(t_tgt, K); Zwin[:, -1] = zhat` when
`NBR_t is not None` — check 3 FAILS while 1 and 2 still pass.

**(iii) The anomaly was already in #414's OWN training artefact, on the OLD sha, with no
roll code involved.** #414 ran at `head_sha` **`8f0e514`** — *the same commit #418 ran*. Its
in-process `t+1` eval (`ml/temporal.py:2011–2048`, `ev_t` keyed on `t_hold[t+1]` over all P
pixels) returned `z_t+1` **0.04369** and `chan_t+1` **0.06548**, against #346's **0.38798**
and **0.23057** — **8.9× and 3.5×**. And the two evals provably scored the SAME points:
`mse_persistence` is bit-identical to sixteen digits in both spaces (**3.139432907104492**
in z, **1.1540813446044922** in channels), which is a pure function of Z, X,
`t_hold`, `K`, `T`, `P` and the seeded RNG draw — so the Z cache, the anomaly transform, the
time holdout and the sample are all pinned identical, and `scale.data_points` confirms the
one thing that did move: **84,405 × 456 = 38,488,680** for #414 against **63,285 × 456 =
28,857,960** for #346, i.e. `ok_p` and nothing else (`temporal.py:1472` is the only line
`pool_x_hold` reaches).

**And the roll AGREES with those training numbers, for both heads, which is the opposite of
what a leaking gather would do.** Window scope, h = 1:

| | roll `msss_pers` h1 | its own training run's `1 − chan_t+1 ratio` | Δ |
|---|---|---|---|
| #422 / #414 head (anomalous) | **0.944** | **0.9433** | **0.0007** |
| #418 / #346 head (control) | 0.811 | 0.8002 | 0.011 |

A roll that injected truth through the neighbours would put #422's h = 1 **above** #414's
own pre-rewrite measurement. It sits ON it — more tightly than the clean control does.

**What that leaves.** The elevated h = 1 is a TRAINING result, not an eval artefact, and it
predates every line of the rewrite. The residual anomaly is the one §7 already isolated —
**error growth over the eleven steps after the first: 1.07× against the control's 1.55×** —
and it cannot be blamed on the rewrite either, because the rewrite does not touch the loop
that produces it. Two of §7's three "impossible" signatures also need softening now that the
control's own audit block has been read: `tau_x` at h = 12 is **0.810 in #418**, on the old
code, rising from 0.753 at h = 1, so "wind stress forecast at a year" is a property of this
`msss_clim` definition on that channel and not something new; and the per-channel gain is
not uniform (`cur_speed` +0.41, `rg_t10` +0.055 at h = 1). The genuinely unexplained
observations are the h = 1 level, the flat error growth, and §6.4's contradiction — **the
same `zhat`, in the same job, gives ORDINARY transport numbers** (`long.r_heldout` 0.469 vs
the control pair's 0.417 / 0.350; `amoc_bands` 0.483 / 0.380 / 0.498 vs 0.476 / 0.355 /
0.437). A rolled state that tracked truth to `acc` 0.97 for twelve months should not project
onto RAPID like an ordinary one.

**#424 was dispatched anyway** — see below. (i)–(iii) are a reading and two measurements on
a toy; #424 is the same question asked of the production artefact, and the record should not
rest on the cheap version of a check when the expensive one costs $1.

**What to check next, in cost order, given the suspect has moved to #414's training:**

1. **The `_trainlon` scope already rules out "the extra pixels were easy."** #414's head
   beats the control on the SAME 22,538 corridor pixels both trained on (0.93933 vs
   0.86700). Whatever happened is not a change in which pixels were scored.
2. **The milestone assets.** #414 published at steps **600 / 60,000 / 120,000**
   (`milestone_steps`). An eval-only re-dispatch of the step-600 milestone would settle
   learning against leakage in one reading: a head 600 steps old cannot have LEARNED a
   one-step error of 0.06 of climatological variance, so if the milestone already shows it,
   the pool change opened a path to the answer rather than to more data.
3. **`train_lon_hold: "none"` on a NON-ring head, or on the gate's architecture.** The
   recipe (`ml/recipes/xl144-nolonhold.json`) differs from #346's dispatch in exactly one
   field, and `pool_x_hold` reaches exactly one line — but it has only ever been run once,
   at one architecture.
4. **#414 has no `probe_head.json`** (its dispatch carried `head_probe: "false"`). The
   unpooled read-out is still the cheapest independent opinion on whether this head knows
   anything the controls do not — ~2 h, ~$0.6, already priced in E-043b §5.

**Not changed here.** `ml/rollout_spatial.py` was read and left alone: the brief was to fix
it only on a proven bug, and the bug is disproven rather than unproven. The only code that
landed is `tests/test_roll_ring_identity.py`.

---

<a id="e-043b-control"></a>
### E-043b-CONTROL · #424 — RESULT, completed 2026-08-20 19:53:47Z. **The CLEAN branch of the pre-registered two-way falsifier fired.** The re-roll of a KNOWN head on the NEW code returns #418's archive to every digit it stores, so #422's anomaly is a property of the HEAD

**#424** re-rolls `e032xl_u1_s0` — **#418's own seed-0 head**, an archived published asset —
beside the frozen `e017_u1_s0` gate, on `gpu-box-31479844` (vast 47724559, the box that ran
both #418 and #422 and still holds their warm Z cache and tensor), at the current `main`.
`window: sroll:e017_u1_s0,e032xl_u1_s0`; **every one of the other 24 inputs is copied
verbatim from #422's own `INPUTS_JSON`**, read out of #422's job log rather than
reconstructed. Both assets were confirmed `state: uploaded` and `HTTP 200` with the exact
bare-`curl` call `sroll_run.sh` makes, before dispatch — the #421 Fastly-404 lesson (§2b),
and cheap where the inputs are all it has cost (§0.3).

**PREDICTION, registered in the dispatch `doc` before the job started, and it is a two-way
falsifier:**

- **If the new code is broken for ring heads**, `e032xl_u1_s0` comes back **flat and
  inflated** — corridor `_trainlon` near 0.94 with a profile that does not decay — instead of
  reproducing **#418's 0.86700** with its normal twelve-month decay (0.774 → 0.650). That
  would locate the fault in `ml/rollout_spatial.py`, retract #422, and make every ring-head
  number written on this sha suspect.
- **If it reproduces 0.86700** within the xl tier's band (§3b pooled sd 0.0021, 95% upper
  bound 0.0037) **with normal decay**, the roll code is CLEAN and #422's anomaly lives in the
  head itself or its published asset — the suspects become #414's training and the #420
  headpub, in the order §9 lists above.

**VOID conditions, also pre-registered:** `e017_u1_s0` must reproduce `horizon_auc` **0.643**
within `GATE_TOL` **0.0101**, and `len(rollout_spatial.json['heads'])` must be **2**.

**First minutes verified, 16:26Z** (§2 — measurements, not intentions). `ml-live-424` is
emitting; the config line reads the frozen anchor (`resumed from run-62.pt at step 60,000`,
`params_M 40.693`, `eval_every 0` — eval-only, nothing trains); **`"heads": 2`**, so both
heads were fetched and the #421 gate-alone failure is ruled out; head 1 of 2 is `s1_s0`, the
gate, stepping at a flat **2.98 s/window**; `gpu_util` on vast 47724559 reads **99%**, so
this is on the card and not on the CPU.

**Head 1 finished and head 2 LOADED, 17:12Z — and the head-2 label is the check that
matters.** The gate completed its 714 steps in **1,989 s**, against #422's 1,988.5 s and
#418's 1,993.0 s: the protocol is reproducing its own wall time to 0.2%. Head 2 then
announced itself as **`s145rspiral:111-4444-0.71-0.5_s0`** — stencil 145, the spiral ring,
seed 0. `rollout_spatial.py` derives that label from the head FILE's own `args`, so it
identifies the ARTEFACT rather than the dispatch tag, and it is the one reading that rules
out #421's failure mode by measurement instead of by a count. Head ETA **10,347 s**, so the
artefact is expected ~**20:00Z**, against `job_timeout` 700 min. **~3.5 h, ≈ $1.0.**

**Harvest instruction, so the reading is not reconstructed later.** Read
`heads['s145rspiral:111-4444-0.71-0.5_s0'].corridor_trainlon` as the mean of its twelve
archived `msss_clim` values (§3b's convention), against **#418's 0.86700**, and read the
twelve values as a PROFILE against #418's `0.774 → 0.650`. Also read
`window.chan_skill[0].msss_pers` against **0.811**, which is where §9's cross-check between
the roll and #346's training-time `chan_t+1` lives. If those three reproduce, the roll code
is clean on the production artefact as well as on the toy, and §9's suspects stand.

**Whatever it returns, §9(ii) already holds independently**: the ring path is bit-identical
to `BASE_SHA` on the toy, and the test that says so fails against the injected bug. #424's
job is to say the same thing about the 211M head, the real tensor and the real Z.

#### HARVEST — completed 2026-08-20 19:53:47Z, 3 h 34 min, ≈ $1.05. **Branch (2) fired: the roll code is CLEAN.**

**Code.** #424 → `head_sha` `ab7a1efe` (`provenance.json` `sha`), job `sroll:` on
`gpu-box-31479844` (vast 47724559), torch 2.13.0+cu126 on an RTX 4090.

[probes-424.json on ml-metrics](https://github.com/blauewelt/earth/blob/ml-metrics/probes-424.json)

[#424 (E-043b-CONTROL re-roll of e032xl_u1_s0 on the new code) — the CI log](https://github.com/blauewelt/earth/actions/runs/32391362292)

**Gate.** `e017_u1_s0` returned `horizon_auc` **0.643**, bands 0.470 / 0.375 / 0.492,
`pass: true`, `fails: []` — the **twenty-second** reproduction of that number (#228 … #422 →
#424), and `len(heads)` is **2**, so neither VOID condition fired.

**The three numbers the harvest instruction named, recomputed to 5 dp from the twelve
archived `msss_clim` values (§3b's convention) rather than read off the rounded field:**

| scope | #424, `e032xl_u1_s0` on the NEW code | #418, the SAME asset on the OLD code (`8f0e514`) | Δ |
|---|---|---|---|
| corridor, blended | **0.68067** | 0.68067 | **0.00000** |
| corridor `_trainlon` (22,538 px) | **0.86700** | 0.86700 | **0.00000** |
| corridor `_holdlon` (7,089 px) | **0.22058** | 0.22058 | **0.00000** |
| `window.chan_skill[0].msss_pers` | **0.811** | 0.811 | **0.000** |

And the profile decays exactly as pre-registered — `0.943 → 0.834` across the twelve leads
(0.943, 0.917, 0.897, 0.883, 0.871, 0.861, 0.852, 0.845, 0.838, 0.832, 0.831, 0.834),
against #422's flat 0.942 → 0.939.

**It is stronger than "within the band", and that is the finding.** §3b would have accepted
anything inside the xl tier's pooled sd of 0.0021. What came back is not a consistent
number, it is **the same record**: walking #418's and #424's head objects key by key, the
ONLY difference in the `e032xl_u1_s0` record is `wall_s` (10,523.7 → 10,521.2) plus the two
fields the new code ADDS (`n_px` and `horizon_auc_daymatched`, the latter equal to
`horizon_auc` in all nine scopes). Every one of the six scopes × twelve horizons, the whole
`audit` block, `amoc_bands` (0.476 / 0.355 / 0.437), `long` (`r_trained` 0.780,
`r_heldout` 0.417, `r_lp18` 0.852) and `future` are byte-identical across the TimeAxis
rewrite. The gate head's record is identical on the same terms (`wall_s` 1,993.0 → 1,991.1).
This is the production-scale version of the toy identity test in §9(ii): 211M parameters,
the real family-3 tensor, the real Z cache, 714 roll steps — and the rewrite moved nothing.

**Therefore the pre-registered conclusion, in its own words:** *"the roll code is CLEAN and
#422's anomaly lives in the head itself or its published asset."* **#422's 0.93733 /
0.93933 / 0.93283 is a property of `e043b-xl144-nolonhold-s0` — the head #414 trained on the
all-longitude pool — and not of `ml/rollout_spatial.py`.** The falsifier that would have
retracted #422 and made every ring-head number on this sha suspect did NOT fire; the
evaluator is exonerated on the artefact as well as on the toy, and §9's suspect list
(#414's training, then the #420 headpub) is what remains.

**What that does NOT settle.** The head could still be anomalous for two very different
reasons — it LEARNED the flat-and-inflated behaviour over 200,000 steps of all-longitude
training (in which case the number may mean something), or the behaviour is STRUCTURAL to
this head class / config / eval path and was there from the start (in which case it cannot
be skill). Those two are separated by **[E-043b-MILESTONE (#425)](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043b-milestone)**,
dispatched immediately below, which rolls the same head at **step 600**.

---

<a id="e-043b-milestone"></a>
### E-043b-MILESTONE · #425 — RESULT, completed 2026-08-21 00:39:08Z. **Branch (2) fired: the anomaly was ACQUIRED, not structural.** Sections 1–5 are the pre-registration written at dispatch; the result is §6

**E-043b-MILESTONE · Rolls #414's OWN step-600 milestone checkpoint beside the frozen
`e017_u1_s0` gate, to separate a LEARNED anomaly from a structural one · params 40.693M
frozen codec (`f3_anchor41M`) + 211.353M head **at step 600** · stage `sroll` (eval-only —
nothing trains; the `steps` input is the codec checkpoint's own 60,000) · data
`family3_na025` (C 39, T 516, sha256 `adcbe700fb6e…`) · arch codec 576×10, 8 heads, d_dec
768, d_z 64, patch 3; head 1024×16, K 24, stencil 145, ring `spiral:111,4444,0.71,0.5` ·
resume `!run-62,run-63` · runner `gpu-box-31479844` (vast 47724559, warm from #418 / #422 /
#424), `job_timeout` 700.**

[#425 (E-043b-MILESTONE, #414's step-600 head rolled) — the CI log](https://github.com/blauewelt/earth/actions/runs/32417515891)

[#425 on the status page](https://blauewelt.github.io/earth/status.html#run-425)

[E-043b-CONTROL · #424, the run that made this the next question](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043b-control)

#### 1 · Why a 600-step head is the discriminator

#424 moved the suspect from the evaluator to the head. A head trained for **600 steps** —
**0.3%** of #414's 200,000-step budget, at a point where the loss curve has barely left
initialisation — **cannot have LEARNED** a one-step error of 0.065 of climatological
variance, and cannot have learned a twelve-month roll whose skill does not decay. So the
step-600 milestone splits the two remaining worlds cleanly, and it does it on a **SHAPE**
(decay versus no decay) rather than on a level, which is why it is readable at n = 1 where
§3b would refuse a level.

#### 2 · PREDICTION, registered in the dispatch `doc` before the job started — two-way

- **(1) IF THE MILESTONE ALREADY ROLLS FLAT AND INFLATED** — corridor `_trainlon` near
  **0.94**, per-horizon `msss_clim` spread over h = 1…12 **under ~0.05** — then the property
  is **STRUCTURAL** to this head class / config / eval path, it was there before any
  training happened, it **cannot be skill**, and **#422's 0.93933 is dead as a skill
  number**. The next suspect is then what stencil-145 ring geometry plus
  `train_lon_hold 'none'` does to the scored quantity, not what 200k steps taught it.
- **(2) IF IT DECAYS NORMALLY** — a young head's profile, falling across leads, corridor
  `_trainlon` far below 0.94 and well below the control's **0.86700** — then the
  flat-and-inflated property was **ACQUIRED** over 200,000 steps of all-longitude training
  and may be a real result. The remaining check is then the **seed-1 replicate of #414**
  (§3b: any new configuration with no measured pair buys its own replication).

**VOID conditions, also pre-registered:** `e017_u1_s0` must reproduce `horizon_auc`
**0.643** within `GATE_TOL` **0.0101**, and `len(rollout_spatial.json['heads'])` must be
**2** (the #421 head-silently-absent failure).

#### 3 · The milestone asset's provenance chain, verified BEFORE dispatch

§0.2 says verify the ARTEFACT, and #382 — a stale Aug-15 leftover published under an E-037
name — is the precedent for what happens when a head's vintage is assumed. **The step-600
milestone was NOT on the release**: `model-checkpoints-v1` carries E-032's milestones
(`e032xlx600_u1_s0__temporal.pt`, `…x60000…`, `…x120000…`, published 2026-08-16) but the
only e043b asset was #420's final `head-weights-e043b-xl144-nolonhold-s0.pt`. Nor could
`scripts/publish_head_weights.sh` have produced it from the box: its `headpub:` source
selector reads only `/opt/earth-cache/ckpt/temporal.pt` or `orphan-temporal-latest.pt`, and
**has no milestone source at all**. So `gpu-box-30257785` was never started, and the file
came from where `ml-train.yml` already puts it — the run's own artifact
(`ml/runs/actions/temporal_ms*.pt`, 30-day retention, uploaded straight off #414's box):

1. **Extracted by HTTP range request from the `probes-414` artifact's zip** (run
   `32278072256`, artifact `9397744228`, 4,651,292,842 B, expires 2026-09-19). The zip's own
   Zip64 central directory names the member `actions/temporal_ms600.pt` at local-header
   offset 3,089,651,439, compressed 777,081,444 B, uncompressed **845,486,131 B**; the
   inflated stream was counted and asserted equal to that. No box, no headpub, no
   intermediate copy — **the stale-leftover path structurally does not exist for this
   asset**.
2. **`torch.load`-ed HERE and its OWN fields read** (not a log line, not the box's print):
   **`step` = 600** and **`run_number` = '414'** — the file states its own vintage and its
   own parent run — with `args` `d_model` 1024, `layers` 16, `K` 24, `stencil` 145,
   `ring_km` `spiral:111,4444,0.71,0.5`, `seed` 0, `input_znoise` 0.0, `steps` 200000,
   `batch` 256, `lr` 1e-3, `train_lon_hold` **`'none'`**, `milestone_steps`
   `'600,60000,120000'`, and **211,352,640 parameters** — equal to #414's `temporal.json`
   `scale.params` exactly, and field-for-field the dispatch #422's audit table already
   pinned for the final head.
3. **Published** to `model-checkpoints-v1` as
   **`head-weights-e043bx600-xl144-nolonhold-s0.pt`** (HTTP 201, asset id 522786521), then
   **polled to `state: uploaded` BEFORE any GET** — the #421 Fastly/`BlobNotFound`
   poisoning lesson, which is a rule about the FIRST request, not about retries.
4. **Round-tripped.** The asset was re-downloaded through the exact release URL
   `sroll_run.sh` builds and hashed: **sha256
   `cce1f3eb5a33c32447d421ae034de46130405a1747b79eab190edce809a9cb3e`, byte-identical** to
   the file that was uploaded, 845,486,131 B. A `headpub` dispatch cannot make this check —
   it uploads and reports an HTTP code — so the artifact route is not merely cheaper here,
   it is better evidenced.
5. **Reachability pre-checked at the two names the script tries**, before the inputs cost
   anything (§0.3): `…-s0__temporal.pt` → **404** (expected; that convention is for full
   checkpoints) and `…-s0.pt` → **206**. Head-load shape is fine —
   `rollout_spatial.py:1569` reads `tk["args"]` and `tk["model"]`, which is exactly what
   `temporal.py`'s milestone save writes.

#### 4 · Dispatch discipline

Every one of the **25 inputs is copied verbatim from #424's own `provenance.json`** except
`window` (`sroll:e017_u1_s0,head-weights-e043bx600-xl144-nolonhold-s0`) and `doc` — §1's
"copy the full INPUTS_JSON block out of the run you are replicating", taken from the
archived artefact rather than retyped. The box `gpu-box-31479844` was started for this and
was **online and idle** at dispatch, so §0e's ordering trap (provision, then let the idle
watch fire into the gap) does not apply — the job took the runner within a minute.

**Expected cost, from #424's own clock:** gate 1,989 s + one xl head ~10,522 s ≈ **3.5 h,
≈ $1.03** at $0.294/h. `job_timeout` 700 min.

#### 5 · First minutes VERIFIED, 21:17Z (§2 — measurements, not intentions)

- The config line reads the **frozen anchor**: `params_M` **40.693**, `resumed from
  run-62.pt at step 60,000` (`parent_tag` `run-80`), `eval_every` **0** — eval-only,
  nothing trains. Steps 12–17 of the job (`Build dataset` → `Train` → `Upload checkpoint`)
  took **2 min 26 s** in total, which is what a no-op training step looks like.
- **`"heads": 2`** in the very first `sroll` record — both the gate and the milestone were
  fetched from the release, so **#421's gate-alone failure is ruled out by measurement**,
  not by a hopeful reading of the asset list.
- The gate is stepping at **2.99 s/window** (120 of 714 in 359 s) against #424's **2.98**
  and #418's 2.79 — the protocol is reproducing its own wall clock again.
- `gpu_util` on vast 47724559 reads **100%**, so this is on the card and not on the CPU
  (§2's four-silently-embedded-on-CPU lesson).
- `eta_all_s` **3,907 s** for the gate + head-2 load; the head itself is the ~10.5 ks that
  follows, so the artefact is expected ~**00:40Z on 2026-08-21**.

**Harvest instruction, written now so the reading is not reconstructed later.** Read
`heads['s145rspiral:111-4444-0.71-0.5_s0'].corridor_trainlon` as the mean of its twelve
archived `msss_clim` values, and read those twelve as a **PROFILE**. The decision rule is
already fixed above: spread over h = 1…12 under ~0.05 with the level near 0.94 → branch (1),
STRUCTURAL, and #422's number dies. A falling profile → branch (2), ACQUIRED, and the arm
goes to a seed-1 replicate of #414. Note that the head label will be
`s145rspiral:111-4444-0.71-0.5_s0` — **identical to the label #422 and #424 print**, because
`rollout_spatial.py` derives it from the file's `args` and this milestone shares #414's
geometry and seed; the run number and the asset name are what distinguish them, so do not
match records across runs on the label alone.

#### 6 · HARVEST — completed 2026-08-21 00:39:08Z, 3 h 34 min 31 s, ≈ $1.05. **BRANCH (2) FIRED. The flat-and-inflated property is NOT structural**

**E-043b-MILESTONE · result · Rolled #414's OWN step-600 milestone checkpoint beside the
frozen `e017_u1_s0` gate · params 40.693M frozen codec + 211.353M head **at step 600** ·
stage `sroll` (eval-only, nothing trained) · data `family3_na025` · arch head 1024×16, K 24,
stencil 145, ring `spiral:111,4444,0.71,0.5` · resume `!run-62,run-63`.**

**Code.** #425 → `head_sha` `1888f642` (`provenance.json` `sha`
`1888f64297878d02a28a2641f2d467f1e1629418`), job `sroll:` on `gpu-box-31479844`
(vast 47724559), torch 2.13.0+cu126 on an RTX 4090. Gate wall **1,993.0 s**, head wall
**10,537.6 s**.

[probes-425.json on ml-metrics](https://github.com/blauewelt/earth/blob/ml-metrics/probes-425.json)

[#425 (E-043b-MILESTONE, #414's step-600 head rolled) — the CI log](https://github.com/blauewelt/earth/actions/runs/32417515891)

**NEITHER VOID CONDITION FIRED.** `e017_u1_s0` returned `horizon_auc` **0.643** (recomputed
from its own twelve `msss_clim` values: **0.64283**), bands 0.470 / 0.375 / 0.492,
`pass: true`, `fails: []` — the **twenty-third** reproduction of that number
(#228 … #424 → #425) — and `len(rollout_spatial.json['heads'])` is **2**. The gate's corridor
scopes are again **0.58908 blended / 0.80425 `_trainlon` / 0.05767 `_holdlon`**, identical to
#417, #418, #422 and #424. Per §3b this measures the PROTOCOL, is not a replicate, and does
not enter the spread column; what it buys is that the numbers below are readable at all. The
head that was scored identifies itself by its own file name in `meta`:
**`head-weights-e043bx600-xl144-nolonhold-s0.pt`** — the step-600 asset, not the final one.

##### 6.1 · The milestone's numbers, recomputed to five decimals from the twelve archived `msss_clim` values (§3b's convention)

| scope | #425, the **step-600** head | #422, the **step-200,000** head | #424 / #418, the E-032 xl144 s0 control | n_px |
|---|---|---|---|---|
| corridor, **blended** | **0.02125** | 0.93733 | 0.68067 | 29,627 |
| corridor, **`_trainlon`** | **0.02967** | 0.93933 | 0.86700 | 22,538 |
| corridor, **`_holdlon`** | **−0.00058** | 0.93283 | 0.22058 | 7,089 |
| window, blended / `_trainlon` / `_holdlon` | **0.03850 / 0.04408 / 0.02383** | 0.93550 / 0.93625 / 0.93375 | — | 84,405 / 63,285 / 21,120 |
| gate scope, blended / `_trainlon` / `_holdlon` | **0.05933 / 0.07033 / 0.02625** | 0.93242 / 0.93350 / 0.92900 | — | 864 / 635 / 229 |

##### 6.2 · The PROFILE, which is what the run was dispatched to read

Corridor `msss_clim`, h = 1 … 12, blended scope:

| head | h1 | h2 | h3 | h4 | h5 | h6 | h7 | h8 | h9 | h10 | h11 | h12 | h1 − h12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **#425, step 600** | **0.179** | 0.101 | 0.062 | 0.040 | 0.018 | 0.006 | −0.003 | −0.009 | −0.012 | −0.023 | −0.036 | **−0.068** | **0.247** |
| #422, step 200,000 | 0.940 | 0.939 | 0.939 | 0.939 | 0.939 | 0.938 | 0.938 | 0.937 | 0.935 | 0.935 | 0.933 | 0.936 | 0.004 |

`_trainlon` decays the same way (0.188 → −0.076, spread **0.264**). The decay is **monotone
at every one of the eleven steps** in both scopes. `acc` falls **0.419 → 0.141** where #422's
is flat at 0.970 → 0.967; `amp_ratio` sits at **0.398–0.457**, i.e. the young head predicts
about 40% of the observed amplitude, where #422's is 0.933–0.954.

**That is branch (2), in the words the dispatch registered before the number existed:** *"IF
IT DECAYS NORMALLY — a young head's profile, falling across leads, corridor `_trainlon` far
below 0.94 and well below the control's 0.86700 — then the flat-and-inflated property was
ACQUIRED over 200,000 steps of all-longitude training and may be a real result."* Corridor
`_trainlon` came back at **0.02967**: 0.91 below #422 and 0.84 below the control, with a
spread of 0.264 against branch (1)'s "under ~0.05" threshold — a **5.3×** margin on the
quantity the decision rule was written on. **Branch (1) — STRUCTURAL, and #422's 0.93933 dead
as a skill number — did not fire and is not close to firing.**

##### 6.3 · Everything else in the record corroborates a genuinely near-initialisation head

| read-out | #425 (step 600) | #422 (step 200k) | #424 (E-032 xl144 s0) |
|---|---|---|---|
| `window.chan_skill[0].msss_pers` | **0.275** | 0.944 | 0.811 |
| `amoc_bands` h1-3 / h4-6 / h7-12 r | **0.323 / 0.319 / 0.194** | 0.483 / 0.380 / 0.498 | 0.476 / 0.355 / 0.437 |
| `long` r_trained (n 195) | **0.040** | 0.790 | 0.780 |
| `long` r_lp18 | **0.050** | 0.876 | 0.852 |
| `long` amp_lp18 | **5.342** | 0.741 | 0.739 |

The 20-year hindcast is the clearest of these: at step 600 the head rolls a smooth
monotone drift with **7.2× the observed low-passed amplitude** and no correlation to truth
(r_lp18 0.050), which is what an untrained autoregressive model does. **`amoc_bands` is the
exception that proves §3b's own caution** — the milestone still returns 0.323 / 0.319 / 0.194
on the three transport bands, numbers that would read as respectable skill if quoted as
levels, from a head with r_lp18 0.05 and corridor AUC 0.021. §3b's band-r row says *"a band r
is a direction, never a level, at any n this programme has"*; this is the strongest single
demonstration of that in the archive, and it is now on the record.

`horizon_auc_daymatched` equals `horizon_auc` in every scope block of this run, as at monthly
it must. `audit.identity_max_dev` **0.0004** (#422: 0.00047).

##### 6.4 · What #425 settles, and what it does not

**SETTLES.** Two of the three worlds the wave has been arguing about are now closed by
measurement rather than by argument:

- **The evaluator is exonerated** (#424): the same asset, the same protocol, the new
  `TimeAxis` code — byte-identical to the old-code record in every scope, every horizon, the
  whole `audit` block, `amoc_bands`, `long` and `future`.
- **The structural explanation is dead** (#425): the property is not in the head class, the
  stencil-145 ring geometry, the `train_lon_hold 'none'` scope definition or the eval path,
  because a head with the identical geometry, identical seed, identical scope split and
  identical eval path — differing only in having taken 600 optimiser steps instead of 200,000
  — rolls at 0.021 with an ordinary decaying profile. **Whatever #422's 0.93933 is, it was
  ACQUIRED over 200,000 steps of all-longitude training.**

**DOES NOT SETTLE — and this is the honest state, not a hedge.** "Acquired" is not
"skill". A 200k-step optimisation can acquire a property that inflates the scored quantity
without forecasting anything; what #425 rules out is only that the property was there before
the optimisation started. Two of §7's four signatures have also **softened under scrutiny**,
and both softenings were found by checking §7's own arithmetic against the control:

- §7(c) called the 39-channel gain **uniform**, and it is not. Recomputed channel by channel
  from `audit.per_channel_msss_clim_corridor`, #422 − #424 at h = 12 ranges from **+0.055**
  (`rg_t10`) to **+0.467** (`rg_s900`), sd **0.105** across the 39 channels; at h = 1 the
  range is +0.024 to +0.255, sd 0.056. The gain rises with lead and its dispersion rises with
  it. "Every channel rises and every channel flattens" is correct; "uniform" is not, and the
  8.5× spread between the smallest and largest channel gain is a shape a mechanism story has
  to explain.
- §7(c) also leaned on wind stress being *"an external forcing field the model cannot
  forecast"*. **#424's own control reads `tau_x` 0.753 / 0.774 / 0.810 at h = 1 / 6 / 12** —
  i.e. the E-032 control, which nobody disputes, already scores 0.81 on `tau_x` at a
  twelve-month lead, and its `tau_x` skill *rises* with lead exactly as #422's does. Whatever
  that says about the corridor metric on a slowly-varying forcing channel, it says about both
  heads, so it cannot be the thing that distinguishes them.

What has NOT softened is §7(a) — the profile itself, 0.004 of decay over twelve months where
the control decays 0.124 and the gate 0.192 — and §7(d), no movement in any transport
read-out. Those two stand exactly as written.

**So the arm's remaining gap is neither the code nor the mechanism: it is `n`.** #422 is one
seed at a configuration no other head in the archive shares, and its number is one that would
headline. §3b names that case three separate times — *"any new metric, cadence, tensor, codec
or scale tier with no measured pair"*, *"any number that will be quoted as a headline in the
paper, whatever its size"*, and *"the first result at a tier buys its own replication"* — and
none of the three has an exception for a direction that looks obvious. The seed pair is
**mandatory**, it was already priced in
[E-043b · the roll §7](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043b-roll)
at ~$4.6 plus its roll, and it went out at 01:12:28Z as
**[E-043b-SEED1 (#426)](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043b-seed1)**.

##### 6.5 · §3b bookkeeping: this run adds NO row to the replicate table

Stated explicitly because §3b requires the table to be extended in the same commit as any new
replicate, and a reader could mistake three matching gate reproductions for three replicates.
**#425 contributes nothing to the spread column.** Its gate is the 23rd run of a fixed
checkpoint through a fixed protocol — §3b's "protocol determinism", a first-class integrity
check and explicitly *not* a replicate. Its milestone head is a **different checkpoint** from
#422's (step 600 against step 200,000), so the pair is not a seed pair and its 0.9081
difference is not a spread. The table's `_trainlon` row therefore still reads **2 pairs,
|Δ| 0.00075 / 0.00108, pooled sd 0.00066 (2 dof)** — which is precisely the yardstick #426
will be read against.

#### 7 · Cost

| item | value |
|---|---|
| box | `gpu-box-31479844` (vast 47724559, Quebec CA, $0.294/h) — warm from #418 / #422 / #424, no embed pass |
| wall | job 21:04:37Z → 00:39:08Z = **3 h 34 min 31 s**; scored wall 1,993.0 s (gate) + 10,537.6 s (head) = 3.48 h |
| money | **≈ $1.05** |
| head roll rate | 10,537.6 s / 714 steps = **14.76 s/step**, inside the 14.7–15.5 band (#394 / #401 / #413 / #417 / #422 / #424) — a step-600 head costs exactly what a step-200,000 head costs, which is worth knowing before the next milestone probe is priced |

---

<a id="e-043b-seed1"></a>
### E-043b-SEED1 · #426 — RESULT (training half), completed 2026-08-21 16:59:11Z. **The one-step ratio REPRODUCES at seed 1 — 0.01400 against #414's 0.01392 — so the training-half anomaly is a property of the configuration, not a single draw. The corridor verdict is #429 and is not in yet.**

Sections 1–6 were written **at dispatch**, hypothesis and falsifier first, so the log could
not be rewritten to fit the answer (§1). They are left exactly as they were written; the
result is §7 onwards, appended when the job landed.

**E-043b-SEED1 · Trains a SECOND xl144 stage-2 head (1024×16, K 24, sunflower-144 = stencil
145) at **seed 1** for 200,000 steps on the EXISTING frozen `f3_anchor41M` codec, with the
stage-2 training pool opened to ALL 481 longitude columns (`--train-lon-hold none`, recipe
`xl144-nolonhold`) — the mandatory seed pair for #414's configuration, which no other head in
the archive shares · params 40.693M codec (frozen) + 211.353M head · stage `stage-2` · data
`family3_na025` (C 39, T 516, sha256 `adcbe700fb6e…`) · arch codec 576×10, 8 heads, d_dec
768, d_z 64, patch 3; head 1024×16, K 24, stencil 145, ring `spiral:111,4444,0.71,0.5` ·
steps×batch stage-1 **60,000 × 512** (= `f3_anchor41M`'s own recorded step count, so NOTHING
trains in stage 1) then stage-2 **200,000 × 256**, expdecay peak 1e-3, halflife 40,000,
warmup 2,000, cooldown-frac 0 · resume `!run-62,run-63` (`f3_anchor41M`, frozen).**

[#426 (E-043b-SEED1, xl144 all-longitude head at seed 1) — the CI log](https://github.com/blauewelt/earth/actions/runs/32435476420)

[#426 on the status page](https://blauewelt.github.io/earth/status.html#run-426)

[plan-426.json — the schedule, published before the run spent anything](https://raw.githubusercontent.com/blauewelt/earth/ml-metrics/plan-426.json)

[E-043b · #414, the seed-0 twin this replicates](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043b)

[E-043b · the roll (#422), the number this exists to replicate](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043b-roll)

#### 1 · Why this run, and why now rather than after more diagnosis

Three runs have narrowed #422's anomaly to one remaining explanation each time, and the
narrowing is finished:

| run | what it ruled out | how |
|---|---|---|
| **#424** (CONTROL) | the evaluator | a KNOWN head re-rolled on the new code returned a byte-identical record |
| **#425** (MILESTONE) | the structural explanation | the same head at step 600 rolls 0.021 with a monotone decaying profile |
| **#426** (this run) | — | nothing is left to rule out by diagnosis; what remains is **n = 1** |

§3b is unambiguous about this case and states it three ways. The seed pair is mandatory for
*any new configuration with no measured pair*, for *any number that will be quoted as a
headline in the paper, whatever its size*, and because *the first result at a tier buys its
own replication*. #422 is all three at once: a training pool no other head in the archive
shares, a corridor `_trainlon` figure that would be the largest in the record, and a single
draw. **"The direction is obvious" is explicitly not an exception.**

#### 2 · PREDICTION, registered in the dispatch `doc` before the job started — two-sided

The reading is a **pair |Δ|** on rolled corridor `_trainlon`, #426 against #422's
**0.93933**, judged against the `_trainlon` pair noise measured on two independent xl pairs:
**#417 |Δ| 0.00075, #418 |Δ| 0.00108, pooled sd 0.00066 (2 dof)** — the tightest scope in the
record — with the xl tier's blended 95% upper bound of **0.0037** as the outer marker.

- **(1) IF |Δ| SITS IN THAT REGIME** — of order 0.001, and in any case inside 0.0037 — then
  0.93933 is a **reproducible property of the all-longitude configuration**, n = 2 at last,
  and the level becomes quotable subject to the two §7 qualifications that survived #425
  (the flat lead-time profile, and no movement in any transport read-out). The mechanism
  question stays open; the *number* stops being a single draw.
- **(2) IF |Δ| IS LARGE** — anywhere near the **0.07179** that separates #422 from its
  control pair, or even at the 0.0224 scale of the 34M tier's worst range — then **0.93933 is
  a single-seed excursion, not a property of the configuration**, the E-043b headline dies
  exactly as E-005's +0.28 did, and the wave's finding reduces to the one clause nothing has
  ever qualified: the `_trainlon` − `_holdlon` gap collapses from ~0.65 to 0.0065, which is a
  statement about WHERE skill sits and is supported by the pixel inventory.

**Both outcomes are results, and both are cheap at this price** (~$4.4 of GPU plus ~$1.1 of
roll). There is no configuration of this run that returns nothing.

**VOID conditions.** The stage-2 checkpoint must record `seed` **1** and `train_lon_hold`
`'none'` with `params` **211,352,640**; and when the roll follows, `e017_u1_s0` must reproduce
`horizon_auc` **0.643** within `GATE_TOL` **0.0101** with `len(heads)` **2**, or no number
leaves it.

#### 3 · Dispatch discipline — 25 fields, 3 changed

§1's rule is to **copy the full INPUTS_JSON block out of the run being replicated** rather
than write the handful of fields the experiment is "about" (the rule that exists because #395
died in 90 s with sixty `size mismatch` lines). The inputs here were read straight out of
**`probes-414.json`'s own `provenance.json`** — the archived artefact, not the log, not a
retype — and exactly three of the 25 fields differ:

| field | #414 | #426 | why |
|---|---|---|---|
| `window` | `…,seed:0,sched:expdecay …` | `…,seed:1,sched:expdecay …` | **the experiment** — one token |
| `runner` | `gpu-box-30257785` | `gpu-box-31479844` | #414's box is stopped; this one was online, idle and warm |
| `doc` | E-043b's | E-043b-SEED1's | §0d |

Everything else is byte-identical, including `head_probe: "false"`. **That is deliberate, not
an oversight repeated.** [#414 §5](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043b)
records the missing head probe as a gap and prices its closure as a separate **eval-only**
dispatch over the published head; flipping it here would make the two seeds
protocol-different on the one axis a paired comparison must hold fixed. The gap is closed for
BOTH seeds at once by that eval, not by making seed 1 a different experiment from seed 0.

**Pre-dispatch checks, all made before the inputs cost anything (§0.3):**

- **The plan was regenerated from the trainer's own scheduler** — `python3
  ml/plan_schedule.py --steps 200000 --lr 1e-3 --schedule expdecay --halflife 40000 --warmup
  2000 --cooldown-frac 0` — and its 260 points are **equal element for element to the
  published `plan-414.json`**. So the curve the status page will draw is provably the curve
  #414 ran, and the `--cooldown-frac 0` that distinguishes this schedule from the script's
  0.1 default was verified by that comparison rather than assumed.
- **The recipe resolves with the seed carried through**: `scripts/resolve_recipe.sh` on the
  exact window string emits `RECIPE_NAME=xl144-nolonhold`, `RECIPE_TRAIN_LON_HOLD=none`,
  `RECIPE_TEMPORAL_D_MODEL=1024`, `RECIPE_TEMPORAL_LAYERS=16`, `RECIPE_STEPS=60000` and
  `WINDOW=stencil:145,…,seed:1,…`.
- **`dispatch_run.mjs` precertified** the plan against the inputs before anything was queued.
- **The runner was online and idle** (`busy=false`) at dispatch and the job took it inside a
  minute, so §0e's ordering trap — provision, then let the idle watch fire into the gap — does
  not apply.

#### 4 · Budget, checked against its own arithmetic rather than copied

§1 says size the job against its own timeout. #414's measured rate is **~247 ms/step** of
stage-2 training, so 200,000 steps is **13.7 h**; its `Probes (K-sweep + stage 2)` step took
**51,317 s** end to end, i.e. 13.7 h of training plus ~32 min of ladder, and its **whole job**
ran 2026-08-19 16:49:09Z → 2026-08-20 07:41:51Z = **14 h 52 min**.

- **`job_timeout` 1800 min (30 h)** — copied from #414 and it does **not** contradict that
  arithmetic: it is 2.0× the measured wall. `job_timeout` is an input, not a cap (§1), and a
  self-hosted runner can run for days.
- **`max_minutes` 0** — also copied, and copied deliberately. E-043e (#415) re-fit its cosine
  **fifteen times** under `max_minutes 1150` and finished at 197,428 steps instead of 200,000;
  `max_minutes: 0` is the only setting that cannot be re-fit, which is what #414 and #416
  used and why their step counts are exact. A seed pair whose two members ran different step
  counts would not be a pair.
- **Expected cost:** 14.9 h × $0.294/h ≈ **$4.38**, plus the follow-on roll below.

#### 5 · The follow-on, so the harvest does not have to re-derive it

**#426 produces no readable number by itself.** Like #414 it trains a head; the corridor AUC
that answers §2 comes from a roll, and the roll needs the head published first. On landing
(**~16:35Z on 2026-08-21** — 200,000 steps from a ~01:25Z stage-2 start at the measured
264 ms/step, plus the ladder and upload; arithmetic, not a reading):

1. **HEADPUB** `e043b-xl144-nolonhold-s1` off `/opt/earth-cache/ckpt/temporal.pt` on this
   box, ~10 min, ~$0.05. Poll the release asset to `state: uploaded` **before any GET** —
   the #421 CDN rule, which is about the FIRST request, not about retries.
2. **`sroll:e017_u1_s0,head-weights-e043b-xl144-nolonhold-s1`** on the same warm box: gate
   ~1,990 s + head ~10,520 s ≈ **3.5 h, ≈ $1.03**, `job_timeout` 700.
3. **Read** `heads['s145rspiral:111-4444-0.71-0.5_s0'].corridor_trainlon` as the mean of its
   twelve archived `msss_clim` values against #422's **0.93933**, and the twelve values as a
   profile against #422's flat 0.940 → 0.936. **Note the label trap** (#425's §5 records it):
   `rollout_spatial.py` derives the head label from the FILE's `args`, and `args['seed']` for
   this head is 1 — so the label will read `…_s1`, unlike the milestone which shared #414's
   seed and printed `_s0`. Match records on the run number and the asset name, never on the
   label alone.
4. **Extend §3b's `_trainlon` row in the same commit as the result**, from 2 pairs to 3.

#### 6 · First minutes VERIFIED, 01:38Z (§2 — measurements, not intentions)

Every line below is a reading off the jobs API, the run's own `ml-live-426` branch or the
Vast API. Nothing is marked passed on an inference.

| # | check | reading |
|---|---|---|
| 1 | **`head_sha`** | **`cb0bdbc2e49aa9ea7e74a8230fa6311195e90c6f`** — current `main`, the sha #425's record was committed on |
| 2 | **`runner_name` is the box, not the `gpu` label** | **`gpu-box-31479844`**, `labels: ["gpu-box-31479844"]`. Picked up **01:12:28Z**, ~14 s after dispatch — no queue-against-idle-runner stall (§2) |
| 3 | **recipe resolved** | stage-1 config line carries **`"recipe": "xl144-nolonhold"`** |
| 4 | **resumed from the anchor, and stage 1 trains NOTHING** | `{"resumed": {"from": "run-62.pt", "parent_tag": "run-80", "at_step": 60000}}` with `params_M` **40.693**, `steps` **60000**, `eval_every` **0** — `while s < a.steps` never turns over. Corroborated by the clock: steps 12–16 ran 01:13:30Z → ~01:17Z, ~3.5 min, which is what a no-op stage 1 costs |
| 5 | **stage-2 config line carries SEED 1** | `{"stage2_config": {"d_model": 1024, "layers": 16, "K": 24, "steps": 200000, "params_M": 211.353, "batch": 256, "train_windows": 38488680, "d_z": 64, **"seed": 1**, "unroll": 1, "stencil": 145, "ring_km": "spiral:111,4444,0.71,0.5", "train_lon_hold": "none", "codec_holdout_lon": "-45,-25"}}` — **the experiment's one changed token, confirmed in the trainer's own record rather than in the dispatch** |
| 5b | **the head is the same head** | `params_M` **211.353** = 211,352,640, equal to #414's `temporal.json` `scale.params` exactly, so `d_z` is 64 and the geometry is not a default-width accident (#395/#387) |
| 5c | **the POOL is the same pool** | `train_windows` **38,488,680** — bit-equal to #414's `scale.data_points`. The two seeds draw from an identical window pool, which is what makes the pair a pair |
| 5d | **the holdout regime is right on both sides** | `train_lon_hold` **`none`** (stage 2 sees all 481 columns) with `codec_holdout_lon` **`-45,-25`** (the frozen anchor's own statistics, untouched) — arm B's design, not arm A's |
| 6 | **LR is non-zero and is the PUBLISHED schedule** | first row: **`stage2_lr` 0.0009999826714706267** at step 2,000. `plan-426.json`'s own point at step 2,000 is **0.000999982671** — the measured rate equals the published curve to nine digits. §2's "a reloaded cosine gave lr 0.0" failure mode is ruled out by measurement |
| 7 | **the run is learning** | step 2,000: `stage2_zmse` **0.8819**, `stage2_val_zmse` **1.14768** against `stage2_monitor.val_persistence` **3.09512** — a val/persistence ratio of **0.371** at 1% of the budget; `stage2_amp` **0.8614**, `grad_norm` **2.3841**. No collapse signature |
| 8 | **it is on the GPU** | `fleet_health` reads **gpu 83%**, cpu 22%, `job=yes` on vast 47724559 — §2's four-silently-embedded-on-CPU lesson |
| 9 | **no embed pass was needed** | stage 2 began ~01:34Z, ~21 min after pickup: the frozen anchor's Z cache and the `family3_na025` tensor were already on this box from #418 / #422 / #424 / #425. #414 paid the same zero, and choosing this box over a cold start saved both an embed and a box-start |
| 10 | **rate, and what it does to the ETA** | **528.2 s / 2,000 steps = 264.1 ms/step**, 7% above #414's 247 ms/step (the first 2,000 steps carry warm-up, so this is an upper bound on the steady rate). Stage 2 began ~**01:25Z**, so 200,000 steps at that rate is **14.7 h** ⇒ training ends ~**16:05Z** and the job, with its ladder and upload, ~**16:35Z**. Far inside `job_timeout` 1800 min. Independently, the status page's own render at 01:37Z reads **"4,000 of 200,000 steps · ~14.5 h left · ends ≈ 16:07"** — the same arithmetic from the other side. Re-time at the next check-in rather than treating 264 as settled |
| 11 | **the STATUS PAGE, screenshotted rather than described** (§2) | `node scripts/status_shot.mjs` at 01:39Z: **`PAGE ERRORS: none`**, 30 runs captured with metrics for #426. The card renders under the **E-043b** tag with the full `doc` string, the generated config line (`params 40.693M codec + 211.353M head · stage stage-2 (temporal head) · data family3_na025 (C 39, T 516) · arch codec 576×10 … head 1024×16 · steps×batch 60,000 × 512, head 200,000 × 256 · resume run-62,run-63 — loaded run-62.pt at step 60,000`), the planned expdecay curve from `plan-426.json`, and a live stage-2 trace reading **z-MSE 0.5190 · held-out z-MSE 0.6643 · amplitude ratio 0.92 · grad norm 1.25 at step 4,000** |

#### 7 · The run, as it happened

Literal readings off the jobs API, the archived `run-426.jsonl` and the run's own step list.
Nothing below is an inference.

| what | reading |
|---|---|
| `run_started_at` | **2026-08-21T01:12:24Z** |
| job wall | 01:12:28Z → **16:59:10Z** = **15 h 46 min** (**56,802 s**) |
| `runner_name` | **gpu-box-31479844** (vast 47724559) |
| `head_sha` | **`cb0bdbc2e49aa9ea7e74a8230fa6311195e90c6f`** |
| conclusion | **success**, and **no step with conclusion `failure`**. Three `skipped`: the GitHub-hosted cache restore (this is a self-hosted runner) and the two joint fine-tune steps |
| step 21 `Probes (K-sweep + stage 2)` | **56,337 s** |
| final progress record | `stage2_step` **200000** · `stage2_zmse` **0.03636** · `stage2_val_zmse` **0.04293** · `stage2_amp` **0.9956** · `stage2_grad_norm` **0.1352** · `stage2_lr` 3.2351468e-05 · `stage2_wall_s` **55512.9** |
| the denominator | `stage2_monitor.val_persistence` **3.09512** (`n_windows` 4096) ⇒ normalised **0.01387** |
| `seed` | **1** — VOID condition cleared |
| `train_lon_hold` | **`none`** — VOID condition cleared |
| `params` | **211,352,640** — VOID condition cleared |
| `stencil` | 145 |
| `train_windows` | **38,488,680** — bit-equal to #414's `scale.data_points`, so the two seeds drew from an identical window pool |

**`val_zmse` was strictly monotone: 100 records carry it, and ZERO of them is an increase.**
First five (step, `val_zmse`): (2000, 1.14768) (4000, 0.66427) (6000, 0.51537) (8000, 0.43305)
(10000, 0.36661). Last five: (192000, 0.04348) (194000, 0.04345) (196000, 0.04312)
(198000, 0.04311) (200000, 0.04293).

**The grad-clip fields are ABSENT and that is CORRECT.** The union of all top-level keys
across the 115 records is exactly `config, resumed, stage2_config, stage2_monitor,
stage2_step, stage2_zmse, stage2_loss_base, stage2_val_zmse, stage2_amp, stage2_grad_norm,
stage2_lr, stage2_wall_s, stage2_probe, stage2_result` — no `stage2_grad_norm_max`, no
`stage2_grad_clip_frac`, no `stage2_grad_nonfinite`. #426 ran the PRE-grad-clip trainer at
`cb0bdbc2`, before `aced980`, exactly as its seed-0 partner #414 / #422 did. That is what
keeps the pair a genuine ONE-VARIABLE seed replicate, and it must be preserved in any future
re-reading: an unclipped seed-1 against a clipped seed-0 would not have been a pair.

#### 8 · The one-step forecast ratio REPRODUCES — and it still is not evidence of skill

This table extends the one in
[E-043b · #414 §3](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043b):

| run | training pool | `mse_model` | `mse_persistence` | **ratio** |
|---|---|---|---|---|
| **#414** (E-043b, seed 0, all longitudes) | ALL 481 cols | 0.0436946 | 3.13943290710449 | **0.01392** |
| **#426** (E-043b-SEED1, seed 1, all longitudes) | ALL 481 cols | 0.04376167 | 3.12627649 | **0.01400** |
| #346 (E-032 xl144 seed 0) — CONTROL | 401 cols, −45..−25 held | 0.3879833 | 3.13943290710449 | 0.12358 |
| #347 (E-032 xl144 seed 1) — CONTROL | 401 cols, −45..−25 held | 0.3797261 | 3.12627625465393 | 0.12147 |

- **Pair |Δ| on the one-step ratio is 0.00008.** Two seeds of the all-longitude
  configuration agree to the fifth decimal.
- **The val draws pair up by SEED, and that is an internal consistency check worth
  recording.** #426's `mse_persistence` **3.12627649** matches #347's **3.12627625465393**,
  and #414's matches #346's to the last bit at 3.13943290710449. Persistence MSE is a pure
  function of the data and the seeded val draw, so each nolonhold head is being read against
  the SAME val sample as its same-seed control. The same-seed contrasts are therefore
  0.12358 / 0.01392 = **8.88×** at seed 0 and 0.12147 / 0.01400 = **8.68×** at seed 1.
- **This does NOT make the 8.7–8.9× a skill result, and §3 of
  [E-043b · #414](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043b)
  already showed why.** The arithmetic there demonstrated that the WHOLE of the seed-0 factor
  is consistent with pure SCOPE: 21,120 of 84,405 ocean pixels (**25.02%**) moved from
  never-seen to trained, and identical trained-pixel skill requires only
  `m_hold(346) − m_hold(414)` = **1.376**. Reproducing at seed 1 confirms that the
  CONFIGURATION reproduces. It cannot confirm skill, because the scope confound is present in
  both members of the pair in equal measure. **The instrument that removes the confound is
  the `_trainlon` corridor AUC, and that is #429's job, not this run's.**
- Recorded without comment beyond the labels: `stage2_result` `chan_mse_model` 0.06428158 /
  `chan_mse_persistence` 1.16594958 = 0.05513; `rapid_r_deseas` **0.496**, `rapid_r_raw`
  0.581, `rapid_r_kfold` **0.476** CI [0.381, 0.564]. #426 carried `head_probe: "false"` by
  design (§3 of this entry), so **there is no unpooled `probe_head` number for either seed**,
  and the pooled `rapid_*` figures above are the legacy-labelled instrument §3 distrusts —
  they are recorded, not quoted.

#### 9 · Artefacts, verified as artefacts rather than as green steps (§0.2)

**`ml-live-426` does not exist.** The run's own step 23 is named "Archive metrics, then clean
up the live branch" and deleted it at 16:59:01–16:59:07Z; that is by design, and a future
harvest should read `ml-metrics`, not the live branch.

On `ml-metrics`: `run-426.jsonl` **23,639 B** (commit `bf56c2ce`, 16:59:03Z),
`probes-426.json` **71,415 B** (commit `526abcca`, 16:59:04Z), `plan-426.json` 9,924 B.
Actions artifacts unexpired: `probes-426` **4,649,445,719 B**, `pixelmae-426`
**424,978,525 B**.

[probes-426.json on ml-metrics](https://github.com/blauewelt/earth/blob/ml-metrics/probes-426.json)

[run-426.jsonl on ml-metrics](https://github.com/blauewelt/earth/blob/ml-metrics/run-426.jsonl)

The 24-hour archive-token failure mode does not apply — the job ran 15 h 46 min — **and it
was checked by the artefacts existing and ending in a `stage2_result` block, not by the
archive step's green tick.**

#### 10 · A NEW infrastructure finding: an xl-tier head is single-copy for its whole run

`scripts/snapshot_head.sh` ran **29 times** across #426, 02:19:36Z → 16:28:42Z, and **every
one failed** with `curl: (22) … error: 422` — the ~2 GiB GitHub release-asset cap against a
~2.4 GB optimiser-carrying checkpoint — each emitting
`##[warning]snapshot: upload of run-426-temporal-latest.pt FAILED — the head is only on this
box`.

So for 15 h 46 min the only copy of 211M trained parameters was
`/opt/earth-cache/ckpt/temporal.pt` on one rented box, and the mechanism built to prevent
exactly that (#396's lesson) was failing silently-but-loudly on every tick.

**This is not specific to #426: any head at this tier exceeds the cap, so every xl run since
the tier existed has been single-copy until its headpub.** The weights-only asset is
**845,487,479 B** and well under the cap — it is only the optimiser state that busts it.

**FOLLOW-UP, not done here:** either strip to weights-only before the snapshot upload, or
route the snapshot to a store without a 2 GiB object cap.

#### 11 · Cost

| item | value |
|---|---|
| box | `gpu-box-31479844` (vast 47724559, Quebec CA, $0.294/h) |
| wall | **15 h 46 min** |
| money | **≈ $4.64** |
| against the estimate | §4 predicted **$4.38**; 6% over, because the measured rate held at ~264 ms/step rather than relaxing to #414's 247 |

---

<a id="e-043b-seed1-roll"></a>
### E-043b-SEED1-roll · #428 (HEADPUB) and #429 (the roll) — DISPATCHED 2026-08-21, in flight. The seed pair §3b requires before any level from the all-longitude configuration may be quoted

Written **at dispatch**, hypothesis first (§1), so the log cannot be rewritten to fit the
answer. §4 was appended as each first-minutes reading landed, and is measurements only.

**#428 · HEADPUB `e043b-xl144-nolonhold-s1` · PUBLISHES the xl144 stage-2 head that #426
(E-043b-SEED1) trained for the full 200,000 steps at SEED 1 on the EXISTING frozen
`f3_anchor41M` codec with the stage-2 pool opened to ALL 481 longitude columns
(`--train-lon-hold none`, recipe `xl144-nolonhold`) · params 211.353M head (+ 40.693M frozen
codec) · stage `headpub` (NOTHING trains; strips `/opt/earth-cache/ckpt/temporal.pt` to
weights-only and uploads `head-weights-e043b-xl144-nolonhold-s1.pt`) · data `family3_na025`
(C 39, T 516, sha256 `adcbe700fb6e…`) · arch codec 576×10, 8 heads, d_dec 768, d_z 64,
patch 3; head 1024×16, K 24, stencil 145, ring `spiral:111,4444,0.71,0.5` · steps×batch
200,000 × 256 stage-2 (= the checkpoint's own recorded count; this job trains zero steps) ·
resume `!run-62,run-63` (`f3_anchor41M`, frozen).**

**#429 · E-043b-SEED1-roll · ROLLS the freshly published seed-1 xl144 stage-2 head
(`head-weights-e043b-xl144-nolonhold-s1`, trained by #426 for the full 200,000 steps at seed
1 on the EXISTING frozen `f3_anchor41M` codec with the stage-2 pool opened to ALL 481
longitude columns) twelve months forward beside the frozen `e017_u1_s0` validation gate,
reading the `*_trainlon` / `*_holdlon` split in every scope — this is the SECOND SEED that
decides whether #422's 0.93933 is a property of the configuration or a single draw · params
40.693M codec (frozen) + 211.353M head · stage `sroll` (eval-only; NOTHING trains, the step
field is the codec checkpoint's own count) · data `family3_na025` (C 39, T 516, sha256
`adcbe700fb6e…`) · arch codec 576×10, 8 heads, d_dec 768, d_z 64, patch 3; head 1024×16,
K 24, stencil 145, ring `spiral:111,4444,0.71,0.5` · steps×batch 60,000 × 512
(= `f3_anchor41M`'s recorded step count, ZERO training steps) · resume `!run-62,run-63`
(`f3_anchor41M`, frozen).**

[#428 (HEADPUB e043b-xl144-nolonhold-s1) — the CI log](https://github.com/blauewelt/earth/actions/runs/32511031975)

[#429 (E-043b-SEED1-roll, the seed-1 xl144 head beside the e017_u1_s0 gate) — the CI log](https://github.com/blauewelt/earth/actions/runs/32511564744)

[#429 on the status page](https://blauewelt.github.io/earth/status.html#run-429)

[E-043b-SEED1 · #426, the training half this publishes and rolls](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043b-seed1)

[E-043b · the roll (#420 + #422), the 0.93933 this exists to replicate](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043b-roll)

[E-043b · #414, the seed-0 twin of the head being rolled](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043b)

#### 1 · #428, the headpub, and the gate it had to pass

#428 completed **success** in ~6 min on `gpu-box-31479844`. `publish_head_weights.sh` printed
verbatim:

```
source: /opt/earth-cache/ckpt/temporal.pt (2.4G)
step=200000 d_model=1024 layers=16 stencil=145 seed=1 znoise=0.0
params=211.4M
weights-only: head-weights-e043b-xl144-nolonhold-s1.pt  807M  (845487479 bytes)
upload … -> HTTP 201
```

**`seed=1` was the whole gate**: it is the only printed field distinguishing this head from
#414's seed-0 head, which matches every other one. `train_lon_hold` is not printed by that
script, so it was verified at source in #426's own job log: `lon holdout · statistics (codec
'-45,-25'): 80/481 cols · training pool (--train-lon-hold 'none'): 0/481 cols`.

Release asset: **845,487,479 bytes, byte-for-byte the same size as its s0 partner**,
`state: uploaded` at **18:03:46Z**. The **#421 CDN rule was observed** — the release JSON was
listed, and **no GET was issued against the asset before `state` read `uploaded`**.

#### 2 · The prediction, restated so the roll cannot be read loosely

The reading is a **pair |Δ| on rolled corridor `_trainlon`**, this run against #422's
**0.93933**, judged against the `_trainlon` pair noise measured on two independent xl pairs
(**#417 |Δ| 0.00075**, **#418 |Δ| 0.00108**, pooled sd **0.00066** on 2 dof), with the xl
tier's blended 95% upper bound **0.0037** as the outer marker.

- **(1) IF |Δ| IS OF ORDER 0.001** — and in any case inside 0.0037 — then 0.93933 is a
  **reproducible property of the all-longitude configuration**, n = 2 at last, and the level
  becomes quotable subject to the two §7 qualifications that survived #425: the flat
  lead-time profile, and no movement in any transport read-out.
- **(2) IF |Δ| IS LARGE** — anywhere near the **0.07179** separating #422 from its control
  pair, or even at the **0.0224** scale of the 34M tier's worst range — then 0.93933 is a
  **single-seed excursion**, the E-043b headline dies exactly as E-005's +0.28 did, and the
  surviving finding is the one clause nothing has ever qualified: the `_trainlon` −
  `_holdlon` gap collapsing from ~0.65 to **0.0065**.

**Both outcomes are results.**

#### 3 · Reading discipline, pre-committed

- **`_holdlon` is quoted as a MECHANISM CONTRAST and never as a level** (§3b: its pair sd
  **0.01079** is **16.4×** `_trainlon`'s, off the identical four checkpoints).
- **Blended-vs-blended is admissible only as an explicitly labelled upper bound**, because
  this head has no `_holdlon` handicap and the E-032 controls' blended figures are deflated
  by a 24% block they never trained on.
- **The label trap** (#425 §5): `rollout_spatial.py` derives the head label from the FILE's
  own `args`, and `args['seed']` here is **1**, so the label will render
  `s145rspiral:111-4444-0.71-0.5_s1` — **match records on the run number and the asset name,
  NEVER on the label alone.**
- **VOID condition:** `e017_u1_s0` must reproduce `horizon_auc` **0.643** within `GATE_TOL`
  **0.0101** with `len(heads)` **2**, or no number leaves the run.

#### 4 · First minutes VERIFIED (§2 — measurements, not intentions)

| # | check | reading |
|---|---|---|
| 1 | **`head_sha`** | **`bfbda006de00cb6deb10b4e7612fb0b3c79c9cdf`** — current `main` |
| 2 | **`runner_name`** | **`gpu-box-31479844`**, job started **18:05:43Z** |
| 3 | **both heads were fetched** | **`"heads": 2`**, read from `ml-live-429`'s `metrics.jsonl`: `{"sroll": {"head": "s1_s0", "head_i": 1, "heads": 2, "phase": "skill"}}`. This is the exact field that caught **#421 (the first E-043b roll attempt, VOID — it rolled the GATE ALONE because the head 404'd)**, and it reads 2, so the run is past `sroll_run.sh`'s hard `exit 1` refusal on a missing head |
| 4 | **measured gate rate** | 160 roll steps in 480 s = **3.00 s/step** ⇒ ~**2,142 s** for the gate's 714 steps, consistent with #422's ~1,990 s |
| 5 | **projected finish** | **≈ 21:40Z**, from roll t0 ≈ 18:11:42Z plus #422's measured gate 1,990 s + xl144 head 10,520 s = 12,510 s ≈ **3 h 28 min**. `job_timeout` **700 min** |

#### 5 · One deviation from the dispatch recipe, recorded because it will bite the next session

**`plan-420.json` and `plan-422.json` do not exist on `ml-metrics`** — nor do `plan-421.json`
or `plan-424.json`. Runs #420 / #421 / #422 / #424 published **no plan at all**, so §1's
"publish the plan" step was silently skipped four times in that wave.

#428's plan was therefore built from **`plan-426.json`** — the curve that actually trained the
head being published, with `note` rewritten to say so — and #429's from **`plan-425.json`**'s
eval shape `{"eval": true, "heads": [...]}` with the head updated to the s1 asset. Both were
precertified clean through `dispatch_run.mjs` and both are live at `ml-metrics/plan-428.json`
and `ml-metrics/plan-429.json`.

Note also that `dispatch_run.mjs`'s eval test is **`temporal_steps == "0"` AND the window
starting with `sroll:`** — so a `headpub:` window falls through to the TRAINING branch and
demands a curve-shaped plan. That asymmetry is what made the substitution necessary, and it
is worth knowing before the next headpub.

#### 6 · Cost

| item | value |
|---|---|
| **#428** (headpub) | ~6 min ≈ **$0.03** |
| **#429** (the roll) | ~3.5 h ≈ **$1.03** |
| box restart | 47724559 was stopped when #426 ended at 16:59Z and was restarted at **17:46:04Z** for this pair, returning fleet burn from **$0.376/h** to **$0.643/h** |

---

<a id="e-043e"></a>
### E-043e · #415 — RESULT, completed 2026-08-20 12:55:52Z. The pentad r2 baseline codec exists and is PUBLISHED; it stopped at 197,428 steps, not 200,000, and it carries no head number

**E-043e · Trains a FRESH 37.976M pentad codec on the r2 tensor (the 39 AMOC channels plus
daily OISST SST as channel 40) with NO longitude holdout — the new PENTAD BASELINE, folding
SST and the corrected holdout regime into one run (recipe `f4r2-40M-nolonhold`, `holdout_lon
'0,0'`) · params 37.976M · stage `encoder` · data `family4_na025_pentad_r2` (C 40, T 3,142,
sha256 `37e146384b6f…`) · arch 512×12, 4 heads, d_dec 256, d_z 32, patch 1 · steps×batch
**197,428** × 512 (dispatched at 200,000; the cosine was re-fit — §1) , cosine 3e-4 to zero ·
resume EMPTY — a fresh codec, `f3_anchor41M`'s 39-row `chan_emb` cannot encode 40 channels.**

**Code.** #415 → `head_sha` `8f0e5141`, job `train` on `gpu-box-39184683` (vast 47724565,
Hong Kong, 504 GB RAM, $0.308/h), torch 2.13.0+cu126 on an RTX 4090.

[#415 (E-043e fresh 38.0M pentad r2 codec, all longitude columns) — the CI log](https://github.com/blauewelt/earth/actions/runs/32278093658)

[probes-415.json on ml-metrics](https://github.com/blauewelt/earth/blob/ml-metrics/probes-415.json)

[E-043 · the wave this arm belongs to](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043)

#### 1 · The recorded step is 197,428, and the arm's own first-minutes check for this FAILED unnoticed

`steps` is the field E-044's dispatch spec calls "the one that can quietly cost a day", so it
was read off the artefact rather than inferred. **The `pixelmae-415` artifact was downloaded
and `torch.load`-ed here** (not on the box, not from the log):

| field | value |
|---|---|
| `step` | **197,428** |
| `args['steps']` | **197,428** — `train.py` assigns the re-fit total back into `a.steps` |
| `tag` | `run-415` |
| `args['holdout_lon']` | **`'0,0'`** — the field `temporal.py` reads to choose the anomaly-transform statistics |
| geometry | `d_model` 512 · `n_layers` 12 · `n_heads` 4 · `d_dec` 256 · `d_z` 32 · `patch` 1 |
| channels | `len(ck['chan'])` **40**, `chan_emb.weight` **(40, 512)** |
| `args['max_minutes']` | **1150** |
| file | 455,915,837 B |

**#415's dispatch raised `max_minutes` from #386's 850 to 1150 for the express purpose of
not being re-fit, and it was re-fit fifteen times anyway.** The doc string says so in as
many words — *"a baseline whose step count is an accident of one box's speed is not a
baseline"* — and it listed **`NO "re-fitting the cosine schedule" line`** as one of four
things to verify in the first minutes (§2). The first re-fit line appears at **18:07:03Z,
one hour and twelve minutes into the job**, at step ~8,199 and 0.31 s/step. It re-fit again
at 10,199 · 12,199 · 14,199 · 16,199 · 18,199 · 24,199 · 28,199 · 30,199 · 34,199 · 50,199 ·
74,199 · 96,199 · 124,199 · 152,199 · 184,199, oscillating between **151,058 and 199,368** as
the measured rate moved between 0.29 and 0.40 s/step, and settling at **197,428**. Nobody
read the check. The lesson is not that 197,428 is wrong — it is a perfectly good codec — but
that **`max_minutes` re-fits whenever the box is slower than the budget implies, and a
larger budget only moves the threshold**; the only dispatch that cannot be re-fit is
`max_minutes: 0`, which is what #414 and #416 used.

The consequence for E-044 is direct and is the whole reason the check exists: a stage-2
dispatch that stated `steps: 200000` against this checkpoint would train **2,572 stage-1
steps**, change the codec weights, change the weight hash, and build a **different Z under a
different cache key** — a job that looks perfect and is not the experiment. **E-044's
`steps` is 197,428.**

#### 2 · PUBLISHED — the codec is now recoverable without a GPU, and the fallback it closes was dangerous

`docs/INFRASTRUCTURE.md` invariant 7 wants every result recoverable from the release, and
#386 spent 14 hours in violation of it. Uploaded 2026-08-20 **14:28Z** to
`model-checkpoints-v1`, both **HTTP 201**, asset `state` polled to **`uploaded`** before any
GET (the #421 CDN rule, §2b of the E-043b roll entry):

| asset | bytes | id |
|---|---|---|
| **`run-415__pixelmae.pt`** | 455,915,837 | 522361772 |
| `run-415__metrics.jsonl` | 23,720 | 522362025 |

**This was more load-bearing than hygiene.** The workflow's "Seed resume checkpoint from the
release" step tries `${TAG}__pixelmae.pt` and then falls back to
**`f3_anchor41M__pixelmae.pt`** — so before this upload, a `resume: !run-415` dispatch onto
any box that did not already hold `/opt/earth-cache/ckpt/run-415.pt` would have silently
seeded the **monthly 39-channel anchor** under the name `run-415`. With the asset present,
the first candidate hits and the fallback is unreachable.

Note also, so nobody mistakes it for this file: **`rescued-orphan-latest-415.pt` on the same
release is NOT #415's output.** It was uploaded at 16:49:32Z on 2026-08-19 — at the START of
#415, by the "Rescue an orphaned checkpoint from the previous job" step — so it is whatever
the previous job left on that box, and it is 64 bytes larger than #415's own checkpoint.

#### 3 · GAP — NO `probe_head.json` / `probe_head_raw3x3.json`. Third time in this wave, same cause

`probes-415.json` carries `probe_kfold`, `probe_sequence`, `dip_check` and `provenance`, and
**no head probe pair**. `provenance.json` records **`head_probe: "false"`** in the dispatch
inputs, because #415's block was copied verbatim from **#386**, which predates the flag —
the same mechanism that cost **#416** its head number and **#414** its unpooled transport
number. §1's "copy the full INPUTS_JSON block" rule worked exactly as written and carried a
stale-but-valid field forward three times in one wave.

**This one matters more than the other two**, because §3's *"at pentad/daily cadence,
spatially POOLED read-outs are distrusted and the HEAD probe is primary"* is scoped to
exactly this cadence. **#415 therefore has no read-out this programme trusts.** Its verdict
does not arrive until a ladder runs with `head_probe: true` over this codec — which
**E-044's own recipe pins** (`"head_probe": "true"` in
`ml/recipes/xl144-zn-pentad-nolonhold.json`, put there for this reason), so the number
arrives with E-044 rather than needing its own dispatch.

#### 4 · The pooled number — recorded, labelled, and NOT a verdict

`probe_kfold.json`, year-blocked k-fold over the frozen codec, section-**pooled** ridge
(`Z.mean(1)` over the ~265-pixel 26.5°N section — the read-out §3 says a mean annihilates):

| | r_kfold_deseas | 95% CI | n | RMSE Sv | σ Sv | lp18 |
|---|---|---|---|---|---|---|
| **#415** — RAPID | **0.637** | [0.578, 0.694] | 1,459 | 3.07 | 3.95 | 0.513 |
| #406 — `f3_anchor41M` on the pentad tensor | 0.660 | [0.593, 0.722] | 1,459 | 2.97 | 3.95 | 0.582 |
| #409 — #386's own r1 pentad codec | 0.652 | [0.582, 0.719] | 1,459 | 3.00 | 3.95 | 0.577 |
| **wind-only baseline** | **0.670** | [0.601, 0.733] | — | 2.93 | — | — |
| **#415** — Florida Current | 0.375 | [0.286, 0.455] | 2,490 | 2.56 | 2.75 | 0.297 |
| #406 / #409 — FC | 0.395 / 0.390 | — | 2,490 | — | — | — |

**Read this as a null, not as a loss.** #415 sits 0.015–0.023 below the two r1 references and
0.033 below wind-only, with CIs that overlap all three across most of their width; the three
are not paired-testable (`probe_kfold.py` still does not dump `pred`/`target_sv`/`years` —
the standing gap from E-043a §3), and pentad cadence has **no measured pair anywhere in the
archive** (§3b). What the table does say is that **neither SST-as-channel-40 nor the opened
longitude pool bought anything visible on the pooled ridge** — which was #415's own
falsifier, *"if its probe ladder does not exceed #386's, neither change buys anything at
pentad cadence"*. **That falsifier is not settled here**, because §3 says the ladder to judge
on is the head probe and this run has none (§3). The head references it will be judged
against, written down now so the comparison cannot be re-chosen later: **anchor head 0.691
[0.631, 0.746] (#406) · #386's own trained head 0.680 [0.617, 0.740] (#409) · raw-3×3 control
0.683 [0.620, 0.742] (both) · wind-only 0.670**.

The other two frozen-codec read-outs, for completeness and used for nothing:

| | #415 | #409 (r1 pentad codec) | #406 (anchor on pentad) |
|---|---|---|---|
| `probe_sequence` K=1 raw / deseas | 0.600 / 0.575 | 0.646 / 0.624 | 0.595 / 0.577 |
| `probe_sequence` K=24 raw / deseas | 0.579 / 0.523 | 0.597 / 0.580 | 0.563 / 0.549 |
| `dip_check` r_oof · sign% · captured% | 0.637 · 72.0 · **31.8** | 0.653 · 72.4 · 40.6 | 0.660 · 71.4 · 45.2 |
| dip predicted Sv (observed −6.95) | **−2.21** | −2.82 | −3.14 |

#### 5 · The training trace — the peak is at 7,500 steps and the run spends the other 190,000 coming back down

From `run-415.jsonl` (28 full probes at `eval_every` 7,500, 13 light probes at
`light_probe_every` 10,000). `linear_r_deseas`, the in-training light probe — §3's
"anything else must be labelled as such", and it is:

| step | 0 | **7,500** | 22,500 | 45,000 | 67,500 | 90,000 | 120,000 | 150,000 | 180,000 | **197,428** |
|---|---|---|---|---|---|---|---|---|---|---|
| `linear_r_deseas` | 0.478 | **0.653** | 0.601 | 0.572 | 0.597 | 0.576 | 0.589 | 0.566 | 0.574 | **0.575** |
| `linear_r_raw` | 0.507 | 0.604 | 0.615 | 0.629 | 0.634 | 0.597 | 0.611 | 0.600 | 0.599 | 0.600 |

**The deseasonalised probe peaks at the FIRST evaluation after step 0 and never returns to
it**: 0.653 at 7,500, then a decline to a plateau of **0.566–0.582** from about step 80,000
onward, dead flat at 0.575 for the last 15,000 steps. The raw probe peaks later and lower in
relative terms (0.642 at 37,500 → 0.600). The reconstruction losses fall normally throughout
(`loss_rec` 0.268 → 0.212, `loss_nei` 0.241 → 0.185) and the forecast diagnostics improve and
then flatten (`z_vs_persistence_pct` 37.8 at 7,500 → **42.1** at the end;
`chan_vs_persistence_pct` 30.6 → **29.0**; the light probe's own
`temporal_r_deseas` 0.585 → 0.529).

**A codec whose reconstruction improves for 190,000 steps while its transport probe declines
is not a new observation** — it is the representation/transport divergence this log has
recorded at every scale — but it has never before been visible this early or this cleanly,
and the probe here is the pooled one, so the decline itself carries the pooled label. Do not
read "train for 8,000 steps instead of 197,428" out of this table; read "the pooled probe
stopped tracking codec quality almost immediately", and let the head probe (§3) settle
whether the same is true unpooled.

#### 6 · Cost

| item | value |
|---|---|
| box | `gpu-box-39184683` (vast 47724565, Hong Kong, **504 GB RAM**, $0.308/h) — built the r2 tensor, holds it warm, and is the ONLY box that has ever touched `family4_na025_pentad_r2` |
| wall | **68,539.9 s = 19.04 h** training, 16:55Z→11:59Z, plus ~44 min of probes and archiving |
| rate | 0.29–0.40 s/step, drifting upward through the run — which is what drove the fifteen re-fits |
| money | **≈ $6.0** |
| dead dispatches charged to this arm | none — #415 ran once, green, first try |

---

<a id="e-043f"></a>
### E-043f · #419 — RESULT, completed 2026-08-21 05:22:58Z, and the verdict is **PROVISIONAL**. The pre-registered falsifier FIRED on both halves — **and every number that fires it is a POOLED one, which §3 distrusts at exactly this cadence.** The daily track is NOT closed by this run

**E-043f · Trains a FRESH 37.976M DAILY codec on ALL 481 longitude columns, so the 45°W–25°W
mid-Atlantic block (25.0% of the domain's ocean pixels) enters stage-1 training instead of
being withheld — the daily arm of the E-043 no-longitude-holdout wave (recipe
`f5-40M-nolonhold`, `holdout_lon "0,0"`) · params **37.976M** · stage `encoder` · data
`family5_na025_daily` (C 39, T 15,706, sha256 `fa2401c248…`, X 165.6 GB fp16) · arch 512×12,
4 heads, d_dec 256, d_z 32, patch 1 · steps×batch **200,000 × 512** · resume **none (fresh —
the `Seed resume checkpoint` step skipped and the trainer started at step 0).**

**Code.** #419 → `head_sha` `19b50368`, job `train` on `gpu-box-46996216` (vast 47913006,
Austria, 700 GB, $0.333/h), torch 2.13.0+cu130 on an RTX 4090. **This is the first daily
codec in the programme's history to reach a probe ladder at all** — #389 (E-038c, the first
daily attempt) wedged 7 h inside the anomaly transform, and #400 and #410 (E-038c, the same
daily codec re-dispatched twice) were cancelled at step 22,000 and step 67,000 of 200,000.

[#419 (E-043f fresh 38.0M daily codec, all longitude columns) — the CI log](https://github.com/blauewelt/earth/actions/runs/32281487754)

[probes-419.json on ml-metrics](https://github.com/blauewelt/earth/blob/ml-metrics/probes-419.json)

[run-419.jsonl on ml-metrics — the complete 242-record ladder](https://github.com/blauewelt/earth/blob/ml-metrics/run-419.jsonl)

[run-419__pixelmae.pt — the published codec, 455,908,925 B](https://github.com/blauewelt/earth/releases/download/model-checkpoints-v1/run-419__pixelmae.pt)

[E-043 · the wave this arm belongs to](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043)

[E-043e · #415, the pentad arm of the same wave, whose dip figure is the comparison below](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-043e)

#### 1 · The pre-registration — it exists, and it is in the dispatch `doc` string, not in this file

Arm F went out at 17:25:14Z on 2026-08-19, after the wave's own entry had been written, and
**no E-043f section was ever opened at dispatch** — so this entry is the first. That costs
nothing here, because the hypothesis and the falsifier were written into the dispatch `doc`
string and survive verbatim in `probes-419.json`'s `provenance.json.inputs.doc`, which is
the artefact rather than anybody's memory of it. Quoted from there:

> **HYPOTHESIS:** the withheld block is a quarter of the ocean and the daily codec is starved
> of it, so returning it lifts the linear section probe out of its flat band and gives it a
> trend.
>
> **FALSIFIER:** a final `linear_r_deseas` inside 0.558-0.598, or flat to within +/-0.02
> across steps 7500-200000 — that would say the daily ceiling is set by cadence or capacity,
> not by the training pool, and the regime change buys the daily arm nothing.
>
> **PRIOR AND CONTROL:** #410 (E-038c daily codec, cancelled at step 67000 of 200000) is the
> withheld-block control at this exact architecture — its `linear_r_deseas` was FLAT in
> 0.558-0.598 with NO trend through step 67000, its last full probe (step 60000) reading
> `linear_r_deseas` 0.582, `linear_r_raw` 0.602, `temporal_r_deseas` 0.627,
> `chan_vs_persistence` 28.1%.

#### 2 · The falsifier FIRED, on both halves — and the control comparison is the cleanest statement of it

`linear_r_deseas` is the in-training linear section probe, and it is **POOLED**: `trainprobe.py`
builds it from `Fsec = Zsec.mean(1)` (`ml/trainprobe.py:336`), a spatial mean over the
26.5°N section. Read every number in this section with that label attached.

**Half one — the final value.** The step-200,000 full probe reads `linear_r_deseas`
**0.570** (pooled). The falsifier band is 0.558–0.598. It fires.

**Half two — flatness.** Across the 40 probes from step 7,500 to step 200,000 (27 full at
`eval_every` 7,500, 13 light at `light_probe_every` 10,000) the pooled trace never leaves
**0.545–0.596**: mean 0.572, sd 0.011, least-squares slope **−0.006 per 100,000 steps** —
flat, and if anything pointing the wrong way. 36 of the 40 sit inside ±0.02 of that mean; the
four that do not (0.593 at 15,000 · 0.596 at 20,000 · 0.549 at 37,500 · 0.545 at 80,000) are
all before step 80,000 and split both ways, and from step 90,000 onward every probe is inside
the band. Over the last 50,000 steps the trace is 0.568–0.574. **There is no trend to find.**

| step | 0 | **7,500** | 20,000 (light) | 37,500 | 80,000 (light) | 120,000 | 150,000 | 180,000 | **200,000** |
|---|---|---|---|---|---|---|---|---|---|
| `linear_r_deseas` (POOLED) | 0.492 | **0.592** | **0.596** (max) | 0.549 | **0.545** (min) | 0.584 | 0.570 | 0.571 | **0.570** |
| `linear_r_raw` (POOLED) | 0.523 | 0.596 | 0.597 | 0.569 | 0.557 | 0.603 | 0.588 | 0.592 | 0.590 |

**The control comparison, which is what the arm was for.** #410 (E-038c daily codec, the
withheld-block run at this exact architecture, cancelled at step 67,000) read `linear_r_deseas`
**0.582** at step 60,000. #419, all 481 longitude columns training, reads **0.576** at step
60,000 and 0.570 at the end. **Returning a quarter of the ocean to the training pool bought
the daily arm nothing on this read-out** — which is the falsifier's own words, and it is the
opposite of what arm B measured at monthly cadence.

The rest of the step-200,000 full probe, each with its pooling label:

| metric | value | pooled? |
|---|---|---|
| `linear_r_deseas` | **0.570** | **POOLED** — ridge on `Fsec = Zsec.mean(1)` (`ml/trainprobe.py:336`) |
| `linear_r_raw` | 0.590 | POOLED (same line) |
| `temporal_r_deseas` | **0.652** | **POOLED** — the mini-transformer's `F[t] = hid[:, -1].mean(0)` (`ml/trainprobe.py:435`) |
| `temporal_r_raw` | 0.662 | POOLED (same line) |
| `chan_vs_persistence_pct` | **+28.4 %** | not pooled — per-pixel channel-space MSE over 8,000 sampled (t, pixel) pairs (`ml/trainprobe.py:389`) |
| `z_vs_persistence_pct` | +26.5 % | not pooled (same sample) |
| `chan_mse_model` / `chan_mse_persistence` | 0.30655 / 0.42822 | |
| `z_mse_model` / `z_mse_persistence` | 11.2086 / 15.25 | |

The step-0 untrained-codec control, which is what says the ladder ran and the training did
something: `linear_r_deseas` 0.492, `linear_r_raw` 0.523, `temporal_r_deseas` 0.565,
`temporal_r_raw` 0.570, `chan_vs_persistence` **−98.2 %**, `z_vs_persistence` +46.5 %. Over
the run `temporal_r_deseas` moves 0.620 → 0.653 and `chan_vs_persistence` 24.6 % → 28.4 % —
**the forecast diagnostics improve monotonically while the pooled transport probe does not
move at all**, which is the representation/transport divergence this log has recorded at
every scale, now visible at daily cadence too.

#### 3 · THE CAVEAT THAT OUTRANKS THE RESULT — there is NO head probe, so §3 distrusts every number above

`probes-419.json` carries `probe_kfold`, `probe_sequence`, `dip_check` and `provenance`, and
**no `probe_head.json` and no raw-3×3 control**. This was not an accident of the run: the
dispatch carried `head_probe: "false"` (recovered verbatim from #389's block, the fourth such
miss in this wave after #414 (E-043b, the xl144 stage-2 head on the all-longitude pool),
#415 (E-043e, the fresh pentad r2 codec) and #416 (E-043a, the monthly f3 codec with no
longitude holdout)), `ml/recipes/f5-40M-nolonhold.json` sets no
`head_probe` key of its own, and the gate that would have run it —
`[ "${RECIPE_HEAD_PROBE:-$IN_HEAD_PROBE}" = "true" ]`, `scripts/probes_run.sh:526` at
`93f1fc2` — therefore never fired.

`ml/CLAUDE.md` §3 is scoped to exactly this cadence: *"At pentad/daily cadence, spatially
POOLED read-outs are distrusted and the HEAD probe is primary."* The reason is mechanical —
`Z.mean(1)` averages over the ~265-pixel 26.5°N section, and geostrophic transport is the
east-minus-west contrast ACROSS that line, the one statistic a mean annihilates. So:

**Every headline number #419 produced is one this programme does not trust, and the null it
appears to report is a null measured on the instrument §3 says not to read.** §2's falsifier
fires on `linear_r_deseas`, which is pooled. §4's k-fold is pooled. §5's K-sweep is pooled.
§6's dip is pooled. **The daily codec has no unpooled read-out and one cannot be quoted.**

**What settles it, and why it is cheap.** The codec is now published as
`run-419__pixelmae.pt` (455,908,925 B, uploaded 2026-08-21 11:51:25Z), so the answer needs
**no retraining at all** — an eval-only ladder with `head_probe: "true"` over that checkpoint,
on the box that still holds `family5_na025_daily`, is the whole cost. That is the named next
action for the daily track, and it is the same shape as the pass E-043a §4(c) still owes
#416 (E-043a, the monthly f3 codec with no longitude holdout).

**And per §3b, this run cannot close the axis even with a head number.** *"Any claim that an
effect is ZERO, or that an axis is CLOSED"* needs two seeds, and *"any new metric, cadence,
tensor, codec or scale tier with no measured pair — the first result at a tier buys its own
replication"* names daily cadence explicitly. This is **n = 1, on a distrusted instrument**.
So the daily track is written here as **UNRESOLVED with a cheap named next step**, not as a
dead end, and nobody should quote "daily buys nothing" as settled on the strength of it.

#### 4 · The pooled k-fold — at daily cadence the codec and a raw wind-stress ridge are indistinguishable on RAPID

`probe_kfold.json`, year-blocked k-fold over the frozen codec. **POOLED**: `F = Z.mean(1)[tidx]`
(`ml/probe_kfold.py:388`), the same section mean §3 distrusts.

| target | r_kfold_deseas | 95% CI | n | RMSE Sv | σ Sv | lp18 | wind-only baseline (same folds) |
|---|---|---|---|---|---|---|---|
| **RAPID** | **0.612** | [0.563, 0.659] | 7,290 | 3.26 | 4.11 | 0.655 | **r 0.607** [0.547, 0.664], RMSE 3.28 |
| Florida Current | 0.364 | [0.287, 0.432] | 13,613 | 2.81 | 3.01 | 0.393 | r 0.110 [0.068, 0.152], RMSE 3.00 |

**On RAPID the codec and the wind-stress ridge are indistinguishable on this read-out** —
0.612 against 0.607, a 0.005 gap with CIs that overlap across almost their whole width. **No
paired test was run**, and `probe_kfold.py` still does not dump `pred`/`target_sv`/`years`
(the standing gap from E-043a §3), so this is two overlapping intervals and not a test —
§3's *"comparing two probes needs a PAIRED test"* is not satisfied and no ordering is claimed.

**On the Florida Current the codec is clearly ahead** — 0.364 [0.287, 0.432] against a
wind-only 0.110 [0.068, 0.152], non-overlapping intervals on n = 13,613. That is the one
comparison in this run where the embedding demonstrably carries something the wind does not,
and it is worth more attention than the RAPID null: FC is the target where the daily cadence
has the most truth to score against.

#### 5 · `probe_sequence` K-sweep and `dip_check`

`probe_sequence.json`, `anomaly_space: true`, n_test 1,095, **POOLED** — the section
embedding it stacks is `emb[t] = z.mean(0)` (`ml/probe_sequence.py:199`), the same spatial
mean under a different name:

| K | 1 | 3 | 6 | 12 | 24 |
|---|---|---|---|---|---|
| r_raw | 0.596 | 0.649 | 0.665 | 0.670 | **0.676** |
| r_deseas | 0.578 | 0.634 | **0.656** | 0.649 | 0.652 |

`seasonal_floor_raw` 0.176 · `seasonal_floor_deseas` 0.084 — both far below the sweep, so the
skill is not the seasonal cycle. **Skill RISES with history and holds** (0.578 → 0.656 by
K = 6, flat thereafter), which is the anomaly-space signature and the opposite of the
state-space failure mode; note that K here counts DAYS, so K = 24 is under a month of context
where the monthly runs' K = 24 was two years.

`dip_check.json` — the 2009–10 collapse, **POOLED** (`F = Z.mean(1)[ridx]`,
`ml/dip_check.py:114`):

Window **2009-09 → 2010-06**:

| | #419 (daily) | #415 (pentad, E-043e) |
|---|---|---|
| `r_out_of_fold` | 0.579 | 0.637 |
| `sign_agreement_pct` | 69 | 72.0 |
| `dip_observed_sv` | −6.84 | −6.95 |
| `dip_predicted_sv` | **−1.75** | −2.21 |
| **`dip_captured_pct`** | **25.6** | 31.8 |

The daily codec captures **a quarter** of the 2009–10 amplitude where the pentad arm captured
a third and the monthly anchor captured half (51.2%, #62 — `f3_anchor41M`, the frozen 40.7M monthly
codec every xl head is built on). All three figures are pooled, none of them is paired, and the two runs differ in cadence AND tensor — so this is a direction, not a
level, and it is recorded because it is the only case-study read the daily track has.

#### 6 · `eval.json` — reconstruction is fine, `t+1` LOSES to persistence, and the NaN defect recurs

Recorded here because **this entry is `eval.json`'s only durable copy**: it was written
inside the `pixelmae-419` artifact (id 9432308132, 293,515,844 B, expiring **2026-09-20**),
and the three files on `ml-metrics` are `plan-419.json`, `probes-419.json` and
`run-419.jsonl` — none of them is it.

Per-channel reconstruction skill, **best**: `rg_t400` 0.919 · `rg_t300` 0.909 · `rg_t200`
0.893 · `rg_t150` 0.862 · `rg_s400` 0.877. **Worst**: `cur_speed` 0.057 · `ssh` 0.075 ·
`tau_x` 0.089 · `tau_y` 0.096 · `rg_s1700` 0.327. The shape is the familiar one — the codec
reconstructs the smooth interior thermal structure and does not reconstruct the fast surface
fields — and at daily cadence the surface channels are the ones that actually move day to
day, which is the mechanism worth carrying into any daily stage-2 design.

**`t+1`: `mse_model` 0.6626 against `mse_persistence` 0.5524 → `beats_persistence: false`.**
On the eval's own one-step test the daily codec is WORSE than predicting no change — while
the in-loop `chan_vs_persistence_pct` reads +28.4 %. The two are different tests (the in-loop
figure is the mini-transformer over K days of context on 8,000 sampled pairs; this one is the
codec's own decoded one-step), and at daily cadence persistence is a far harder baseline than
at monthly — one day of ocean change is small. Both are recorded; neither is reconciled here.

**The NaN defect fired again, unguarded.** `rapid_probe` in `eval.json` reads `pearson_train`
**NaN** and `pearson_heldout_years` **NaN** with n_train 6,195 and n_test 1,095. That is
`docs/INFRASTRUCTURE.md` §4 invariant 12 (*"a result file is never written containing NaN —
the job stops instead"*) and `ml/CLAUDE.md` §5.22 (*"Never write NaN into a results file"*),
and it is the **same defect, in the same file, that #386 (E-038a, the f4-40M pentad control)
hit** — where the log already noted the guard did not fire. **Recorded as a defect
recurrence**: the guard named after #386 still does not exist, so a second run has now
written NaN into a results file and finished green. `ml/probe_sequence.py:203` shows what
the fix looks like — it refuses and exits rather than writing a non-finite array — and
`eval.json`'s `rapid_probe` has no such check.

#### 7 · Cost — and the finding hiding inside it: at daily cadence the INSTRUMENT costs more than the EXPERIMENT

| item | value |
|---|---|
| job wall | 2026-08-19T17:25:18Z → 2026-08-21T05:22:58Z = **35.96 h** on `gpu-box-46996216` (vast 47913006, RTX 4090, $0.333/h) |
| money | **≈ $12** |
| `Train` step | 17:33:11Z → 02:39:56Z = **33.11 h**; the trainer loop's own `wall_s` is **115,729.0 s = 32.15 h** |
| `Probes (K-sweep + stage 2)` step | 02:40:57Z → 05:22:47Z = **2.70 h**, on top of the loop |
| training only | **43,479.1 s = 12.08 h** ⇒ **217.4 ms/step** — identical to #410's measured 217.4 ms/step on this same box |
| **in-loop probes** | **72,249.9 s = 20.07 h = 62.4 % of the loop** — 28 full probes (27 at 2,275.9–2,307.0 s plus the step-0 control at 2,316.6 s) and 13 light probes at 604.9–618.4 s |
| dead dispatches charged to this arm | none — #419 ran once, green, first try. (#389, #400 and #410 — the three cancelled daily attempts — are charged to E-038c, not here.) |

**Twelve hours of training bought twenty hours of measurement.** `eval_every` 7,500 and
`light_probe_every` 10,000 were copied from #389's block (E-038c, the first daily attempt),
where they were chosen for a run that never finished; at daily cadence a full probe costs ~38 min against 217 ms/step of
training, so the settings that are merely generous at monthly cadence dominate the bill here.
**Any future daily dispatch should cut `eval_every` and `light_probe_every`** — halving the
full-probe count alone would have returned ~9.5 h and ~$3 with no loss to the ladder above,
whose 40 points are flat to sd 0.011.

#### 8 · The archive steps ran past GitHub's 24-hour token ceiling, failed with 401, and REPORTED SUCCESS

`ml/CLAUDE.md` §0.2 (*"a step that reports success is not evidence it did anything"*) and §4.6
(*"a step that can fail silently will"*) — the most expensive instance of that pair this
programme has recorded, because it applies to every long run rather than to one bad step.

An Actions job token hard-expires at **24 h**. #419 ran **35.96 h**, so by the time step 22
`Upload probe results` and step 23 `Archive metrics, then clean up the live branch` executed
at 05:22:47–05:22:54Z, the `GITHUB_TOKEN` was almost twelve hours past its ceiling. Both failed with `401 Bad
credentials`; both **report `success` in the jobs API**, along with every other executed step
in the job. The 401s exist only in the log text (`##[warning]metrics archive failed — live
branch kept for recovery`, `##[warning]archive failed: 401 Bad credentials`,
`##[warning]probe archive failed — results are only in the artifact, which expires`).

**What that cost.** The archive steps landed **nothing**, and the hand rescue that followed
inherited the failure: the copy pushed to `ml-metrics` at 05:57:07Z on the 21st
(`1b6e26ec`) was taken from the **DEAD `ml-live-419` branch**, whose publisher had itself
died at 2026-08-20T17:21:31Z — **172 records, last line `step 142000`**, while the run went
on to 200,000. For the next six hours the only public copy of a 36-hour result stopped 58,000
steps short of its own ending, and the complete log plus the 455.9 MB codec existed **only
inside an Actions artifact expiring 2026-09-20**. Both are now published: the complete
242-record `run-419.jsonl` on `ml-metrics` (`cd1bee82`, 11:50:29Z), and
`run-419__metrics.jsonl` + `run-419__pixelmae.pt` on the `model-checkpoints-v1` release
(11:51Z). Nothing was lost, and nothing about that was designed — **a truncated file with no
error in it is exactly the shape §0.2 warns about**, and it looked complete.

**The durable fix is still outstanding, and it is one line of workflow:** any job that can
exceed 24 h needs a **PAT on its archive steps**, not the automatic job token. Every daily-arm
training is in that class by construction. Until it lands, harvest every long run by hand and
**never read a green archive step as evidence that anything was archived**. Filed as an open
follow-up in `ml/CLAUDE.md` §8.

---

<a id="ops-2026-08-20-1445"></a>
## OPERATIONS · 2026-08-20 ~14:45Z — E-044 was blocked for 36 minutes on a Vast host with no free GPU. **RESOLVED at 15:07:23Z; #423 went out at 15:10:05Z**

> **SUPERSEDED IN PART, 15:10Z.** `gpu-box-39184683` came up on the **28th consecutive start
> attempt at 15:07:23Z**, the runner registered `online` / `busy: false`, and
> **[E-044 · #423](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-044)**
> was dispatched at 15:10:05Z with `steps: "197428"`. §1 below is therefore history — but it
> is left standing, because the *reasoning* in §2 (why no other box was substituted) and the
> credit arithmetic in §4 are unchanged and are what the next session needs. **§4's "if E-044
> also starts" column is now the live case.**

Not an experiment. Recorded because a session spent 36 minutes unable to fire a prepared
dispatch, and the next one must not have to re-derive any of it.

### 1 · The block, measured

**`gpu-box-39184683` (vast `47724565`, Hong Kong, RTX 4090, 504 GB RAM, 100 GB disk,
$0.308/h) is `exited`, and its host has no free GPU.** Every start attempt since **14:31Z**
returns

```
{"success":false,"error":"resources_unavailable",
 "msg":"Required resources are currently unavailable, state change queued."}
```

and the instance's `next_state` stays `stopped` — the "queued" in that message has not
produced a queued state change on the object. Retried continuously since 14:31Z. This is
the failure §4 of the dispatch spec named after **#407**.

**RESOLVED at 15:07:23Z** — attempt 28 returned `{"success":true}` and the box is `running`
with its runner `online` and `busy: false`. **It is host-specific, not account-level, and
that was checked rather than assumed:**
`47720664` (Brazil, 126 GB RAM) started **`{"success":true}`** on the first attempt at
14:44Z. It was **stopped again within three minutes** (~$0.015) because idle burn is stopped
on sight (§7); it was started as a diagnostic and for no other reason.

**The box is STOPPED, not destroyed.** Its disk — `/opt/earth-cache/ckpt/run-415.pt` and
`ml/cache/family4_na025_pentad_r2.npz` — is intact. **Do not destroy it.**

### 2 · Why E-044 waits for that box specifically, now that one of the two reasons is gone

Spec §4 gave two reasons and this session removed the first one:

- **`run-415.pt` existing only on that disk — FIXED.** `run-415__pixelmae.pt` is on
  `model-checkpoints-v1` (§2 of the E-043e entry). `resume: !run-415` is now satisfiable on
  any box, and the dangerous `f3_anchor41M__pixelmae.pt` fallback is unreachable.
- **The warm tensor — STILL BINDING, and for a scientific reason rather than a cost one.**
  `family4_na025_pentad_r2` is **not published anywhere**: the data release carries
  `family3_na025` and the raw inputs only, and `gpu-box-39184683` is the **only box that has
  ever touched it** (#405, #408, #411, #415). Any other box runs the family-4 branch of the
  build step — a Hub fetch of `pentad025/{index,pentad_mean_uo,_vo,_mlotst,_zos}`,
  `truth_pentad.npz` and the 4.25 GB SST artifact, then `build_family4.py` writing its own
  33 GB — and **a box that builds its own tensor is the E-008 box effect**, measured at
  **0.041 on the head k-fold and 0.111 on the 36-month split** at a fixed seed. Published Z
  removed that cause; rebuilding the tensor reinstates it, on the **first stage-2 run at a
  new cadence**, where §3b says every number is n = 1 already. A cheap-looking box swap
  would buy a contaminated result.

**So the honest options, in order, and none of them is "dispatch somewhere else today":**

1. **Wait for `47724565`.** Free, and it is the only path that produces the experiment as
   designed. Retry `node scripts/gpu_box.mjs start 47724565` until it returns
   `{"success":true}`. **THIS IS WHAT HAPPENED** — 36 minutes and 28 attempts, at ~55 s
   apart. Cost of waiting: nothing. Cost of not waiting: a contaminated n = 1.
2. **PUBLISH THE PENTAD TENSOR** — the real fix, and it is a work item rather than a
   dispatch. `scripts/data_release.mjs` already streams a split tar to the release and
   `family3_na025_adcbe700fb.npz.{aa,ab}` is the pattern; `family4_na025_pentad_r2.npz` is
   4.5 GB, i.e. **three chunks**. It can only be done **from that box**, so it is blocked on
   option 1 too — but it should be done the moment the box is up, before the training job
   takes the machine, because it permanently removes this whole class of block **and** it is
   a hard prerequisite for the §7a pentad `sroll:` (which needs the tensor on whatever box
   rolls it).
3. **Rebuild elsewhere and accept the box effect** — rejected above. If it is ever taken,
   the rebuilt tensor's sha256 must be compared against **`37e146384b6f622fefe3c7e18ad9bab0389c9538be79536899fe8729bb2d0826`**
   and a mismatch reported as a finding, not worked around.

### 3 · The dispatch, built and validated — fire it verbatim

`steps` is **197,428**, read off the `pixelmae-415` artifact (E-043e §1), **not** the 200,000
the spec's §3 expected — #415 carried `max_minutes 1150`, not 0, and was re-fit fifteen
times. All 25 inputs are present and their names were checked against
`.github/workflows/ml-train.yml`'s own `workflow_dispatch.inputs` block (25/25, no extras,
no omissions); `python3 tests/test_workflow_config.py` passes 5/5; the JSON is 4,672
characters against the 21,000 ceiling; `bash scripts/resolve_recipe.sh` reproduces the whole
`RECIPE_*` block of spec §1 including `RECIPE_HOLDOUT_LON=0,0`, `RECIPE_TRAIN_LON_HOLD=none`
and `RECIPE_HEAD_PROBE=true`.

The exact input set (the `doc` string carries the twelve-item first-minutes checklist and is
elided here for width — **the complete 25-field JSON, verbatim and ready to paste, is in the
project doc `claude/E044-dispatch-READY.md`**, together with the artefact read-out that
produced `steps` and the first-minutes checklist):

| field | value | | field | value |
|---|---|---|---|---|
| `steps` | **`197428`** | | `tensor` | `family4_na025_pentad_r2` |
| `batch` | `512` | | `sst_channel` | `false` |
| `d_z` | `32` | | `patch` | `1` |
| `anomaly` | `true` | | `max_minutes` | **`0`** |
| `temporal_steps` | `200000` | | `runner` | **`gpu-box-39184683`** |
| `temporal_d_model` | `1024` | | `job_timeout` | `2400` |
| `temporal_layers` | `16` | | `lr_floor` | `0` |
| `eval_every` | `0` | | `lr_decay_steps` | `0` |
| `resume` | **`!run-415`** | | `codec_d_model` | `512` |
| `head_probe` | **`true`** | | `codec_layers` | `12` |
| `light_probe_every` | `0` | | `codec_heads` | `4` |
| `window` | `recipe:xl144-zn-pentad-nolonhold,stencil:145,ring:spiral:111-4444-0.71-0.5,seed:0,sched:expdecay --lr-cooldown-frac 0 --milestone-steps 600,60000,120000 --input-znoise 0.7` | | `codec_d_dec` | `256` |

```
node scripts/gpu_box.mjs start 47724565          # until it returns {"success":true}
node scripts/fleet_dispatch_wf.mjs '<the JSON in claude/E044-dispatch-READY.md>' main
node scripts/publish_plan.mjs <n> '{"steps":200000,"lr":1e-3,"schedule":"expdecay","halflife":40000,"stage":"stage-2"}'
```

`max_minutes: 0` is not decoration. #415 is the second run in three days whose cosine was
re-fit by a non-zero `max_minutes` (#386 was the first, 200,000 → 166,752), and a re-fit here
would move stage 2's own step budget rather than stage 1's. **The only budget that cannot be
re-fit is 0**; `job_timeout 2400` is the cap that actually stops the job.

### 4 · Credit — MEASURED, and the fleet outlives it by hours in every scenario

**Measured directly off the Vast account, three samples across 24 minutes** — the ledger
does not update continuously, so read the endpoints and not the middle:

| at | credit | implied burn since the previous sample |
|---|---|---|
| 2026-08-20 14:26Z | **$5.8854** | — |
| 2026-08-20 14:43Z | **$5.7219** | $0.564/h |
| 2026-08-20 14:50Z | **$5.63** | $0.78/h |
| **14:26Z → 14:50Z, the honest endpoint-to-endpoint figure** | **−$0.2554** | **$0.639/h** |

`scripts/publish_fleet_status.mjs`'s own formula — running boxes' `dph_total` plus stopped
boxes' `storage_total_cost` — reports **$0.711/h** and **7.92 h** of runway at 14:50Z, which
brackets the ledger figure from above. **Take $0.64–0.71/h and a runway of 7.9–8.8 h:
exhausted between ~22:45Z tonight and ~00:00Z.** (The comment in that script records the
opposite bias on 2026-08-19 — $1.571/h actual against $1.256/h by the formula — so neither
estimator is trusted alone; both are quoted.)

One running box (`gpu-box-46996216`, $0.333/h, #419) plus storage on twelve stopped
instances — storage is charged whether an instance runs or not, and at ~$0.38/h by the
`storage_total_cost` sum it is **more than half the bill while producing nothing**.

| | |
|---|---|
| credit now (14:50Z) | **$5.63** |
| burn now (1 box) | **$0.64–0.71/h** → exhausted **~22:45Z–00:00Z** |
| burn with E-044 running (2 boxes) | ~$0.95–1.02/h → exhausted **~20:20Z–20:45Z TONIGHT** |
| **#419's tail** — 73,000 steps at 0.60 s/step | 12.2 h, ETA **~02:50Z**, needs **~$4.1** of GPU |
| **E-044** — ~30 h (embed ~10 + train ~15 + probes ~2.5 + ladder ~2.5) | needs **~$9.2** of GPU |
| storage over that window | ~**$6.9** |
| **to finish both** | **≈ $20** ⇒ **≈ $15 MORE IS NEEDED TODAY** |
| including E-044's follow-on pentad `sroll:` (~14 h, spec §7b) | ≈ $24 ⇒ **≈ $19 more** |

**Without a top-up: #419 dies at roughly step 176,000–184,000 / 200,000 between 22:45Z and
midnight** — three to four hours and 16,000–24,000 steps short of a daily codec that has been
running for 33 hours — **and E-044, if it starts, dies inside its embed pass having produced
nothing at all.** Per §0e the fleet is not
being parked for this and nothing has been scaled down to fit; the arithmetic is here because
that is what §0e asks a session to owe instead.

**One lever that costs nothing and is worth taking:** eleven of the twelve instances are
stopped and eight of them have no unique warm state left worth keeping. Destroying the ones
that do not (**keeping `47724565` — the pentad tensor and `run-415.pt`; `47724559` — #417/
#418's warm family-3 Z; `46996216` — #419, running; `30257785` — #414's `temporal.pt`**)
would cut ~$0.15/h of pure storage, ~27% of the current burn, and buy roughly 2.5 extra hours
of #419. Not done here, because it trades a runway hour against a warm cache and that is
Chris's call, not a night-shift session's.

### 5 · #419 — one-line health

**#419 (E-043f fresh 38.0M daily codec, all longitude columns)** on `gpu-box-46996216`
($0.333/h, Austria): **step 127,000 / 200,000** at 14:35Z, `phase` `training`, GPU **97%**,
CPU 5%, disk 52%, `loss_rec` **0.236** / `loss_nei` **0.187**, 0.60 s/step averaged over the
whole run from the live branch's own `wall_s` — **~12.2 h remaining, ETA ~02:50Z on
2026-08-21**, which is ~2 h past the credit. Healthy; nothing to do but feed it.

[#419 (E-043f daily codec) — the live status page](https://blauewelt.github.io/earth/status.html#run-419)

---

<a id="wave9-status-2026-08-19-2055"></a>
## OPERATIONS · The E-043 wave at 20:55Z — one arm home, four running, runway shorter than two of them

Not an experiment. Recorded because it sets what the next session inherits.

| run | arm | progress at 20:55Z |
|---|---|---|
| **#414** (E-043b xl144 stage-2 head, all-longitude pool, frozen anchor) | B | stage-2 step **52,000 / 200,000** |
| **#415** (E-043e fresh 38.0M pentad r2 codec, all longitude columns) | E | step **37,000 / 200,000** |
| **#416** (E-043a monthly f3 codec, no lon holdout) | A | **DONE** 20:46:10Z — box stopped |
| **#417** (E-043d sroll re-roll, gate + xl233 pair) | D1 | sroll head **3 / 3**, phase `skill` |
| **#418** (E-043d sroll re-roll, gate + xl144 pair) | D2 | **queued** behind #417 on the same box, by design |
| **#419** (E-043f fresh 38.0M daily codec, all longitude columns) | F | step **12,000 / 200,000** |

**Money.** Credit **$30.60**, burn **$1.256/h** after `gpu-box-46045353` was stopped on
#416's drain, runway **~24 h**. **#415 and #419 both run past it** — #415 needs ~163k
more steps and #419 ~188k. Chris's instruction is to **count on a top-up rather than park
the fleet**, so the fleet is not being triaged the way it was at
[01:35Z](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#credit-triage-2026-08-19);
the standing lesson from that night still applies — an exhaustion that stops N boxes at
once produces N zero results, not N partial ones.

**Open on this wave, in order of cheapness:**

1. `probe_kfold.py` does not dump `pred`/`target_sv`/`years`, so **no two pooled-ridge
   k-folds in this programme can be paired-tested**. ~2 KB per target, arrays already in
   memory. E-043a §3.
2. An **eval-only** ladder with `head_probe: true` over #416's codec — the head number
   #416 never took, ~2 h, ~$0.6. E-043a §4(c).
3. Arm **F (daily)** — **#419 (E-043f fresh 38.0M daily codec, all longitude columns)**
   — is the arm the 17:2xZ pass refused, because `holdout_lon` is a recipe-only key and
   no daily recipe existed on main. `f5-40M-nolonhold` landed and it went out at
   17:25:14Z onto `gpu-box-46996216`, which had been idle-burning at $0.333/h. It is the
   longest arm in the wave and the one most exposed to the runway.

---

<a id="holdout-lon-band-2026-08-19"></a>
## §0d · DEFECT & FINDING · The skill map's central band is the held-out longitude block, 2026-08-19 ~14:30Z

Not an experiment — a diagnosis, made from artefacts already on disk, that
changes how every rolled number in this log should be read. It costs no GPU and
it was available to be made at any point since E-022.

**The finding.** `ml/paper/fig_gulfstream`'s dominant feature — a meridional
band of collapsed skill through the central Atlantic — is **not an ocean
feature and not a model failure in the ocean**. It is the longitude block
`45°W–25°W`, which is withheld from training in **both** stages and then scored
by the rolled evaluation along with everything else. The band is the model's
**spatial generalisation gap**, and the map is a direct picture of one.

**Verified in source, not inferred:**

| where | what it does |
|---|---|
| `ml/train.py:119` | `p.add_argument("--holdout-lon", default="-45,-25")` |
| `ml/train.py:288` | stage-1 codec pool = `obs_any & ~t_hold[:,None,None] & ~x_hold[None,None,:]` — line 289 puts the same block in *validation* |
| `ml/temporal.py:1180-1181` | stage 2 reads `lo, hi` from the **frozen codec's** `ck["args"]["holdout_lon"]` and rebuilds `x_hold = (lons >= lo) & (lons < hi)` — half-open, so the edges are exactly −45.00 and −25.00 |
| `ml/temporal.py:1419` | stage-2 head pool = `ok_p = ~x_hold[xs]` |
| `ml/rollout_spatial.py:566-567` | recomputes the identical `x_hold` from the checkpoint's args … |
| `ml/rollout_spatial.py:630` | … uses it **only** for `stream_stats`' normalisation constants, and then scores all three scopes over every pixel, recording nowhere in the artefact that it did |

**The mechanism is explicit.** `temporal.py:826` builds
`coords = np.stack([lats[ys]/90, lons[xs]/180], 1)` and lines 1343/1351
concatenate it into `static_ctx`. **Longitude is a literal input feature with a
20° hole in its training range.** The head is interpolating across a gap in its
own input, which is why the failure is a smooth ramp rather than a cliff.

**Evidence, all recomputed this session from `ml/paper/roll_356.json` +
`data/amoc_eval_mask.json` (xl144 seed 0, h = 6):**

- **Edges at exactly −45.00 / −25.00.** The earlier reading of "42°W–28°W,
  near-vertical" was a **2.5°-binning artefact** — corrected. The edges are
  **ramps 1–2° wide**.
- **Monotone decay inward from BOTH trained edges**, meeting in the middle:
  in from 45°W at 1° steps 0.702 / 0.485 / 0.341 / 0.233 / 0.186; in from 25°W
  0.708 / 0.540 / 0.421 / 0.329 / 0.263; block centre (−36…−34) **0.171**.
- **The ten steepest column-to-column gradients in the whole 481-column
  window** all lie inside the block and within 2.25° of an edge. Nothing else
  in the basin has a gradient like it. This is now **asserted** in
  `ml/paper/make_figs.py`, so a future rollout that moves the holdout fails the
  figure build instead of quietly drawing a mislabelled band.
- **Latitude-invariant** (in/out by 10° band): 0–10N 0.290/0.834 · 10–20N
  0.295/0.858 · 20–30N 0.436/0.849 · 30–40N 0.306/0.835 · 40–50N 0.157/0.865 ·
  50–60N 0.386/0.900 · 60–70N 0.424/0.910. That **kills** the Mid-Atlantic-Ridge
  and the Argo/observing-coverage readings, both of which are latitude-structured.
- **The variance explanation is REFUTED by measurement.** MSSS is a
  variance-normalised ratio, so the previous draft blamed a small denominator.
  Per-pixel standardised-anomaly variance computed from `ml/cache/base025_na.npz`
  on the eval's own recipe (train-month climatology, holdout years excluded,
  per-channel σ over train months × non-holdout lons) is **smooth and edgeless**
  through the region: −52 1.27 · −48 1.29 · −45 **1.38** · −42 1.42 · −40 1.24 ·
  −36 0.99 · −30 0.83 · −25 **0.79** · −22 0.77 · −16 0.74. **Nothing happens at
  −45 or −25.** Variance is nearly 2× larger at the west edge than the east
  edge, where skill is equal to 0.01 (0.781 vs 0.772); the zonal minimum is at
  **12°E**, far east of the band; and `corr(variance, skill)` over the window is
  **−0.05**. Denominator size does not order skill. *(Caveat: `base025_na.npz`
  carries 3 dynamic channels — `cur_speed`, `log_mld`, `ssh` — not the eval's
  39. The shape of the profile is the claim; the absolute level is a proxy.)*
- **Not the codec.** E-019b1 measured the retrained deep-T decoder's fidelity
  at **1.43% rmse² on held-out longitudes vs 0.85% on trained ones** (against
  1.90% on held-out *months*) — the codec generalises across the block
  essentially perfectly. The collapse is **entirely a stage-2
  forecast-head generalisation failure.**

**Overlap with the published scopes**, measured off `data/amoc_eval_mask.json`:

| scope | in-block / total | share |
|---|---|---|
| rolled window | 21,120 / 84,405 | **25.0%** |
| AMOC corridor | 7,089 / 29,627 | **23.9%** |
| RAPID 26.5°N section | 80 / 265 | **30.2%** |

**Direction is DEFLATIONARY — nothing in this archive is inflated.** Including
the block makes every published number *worse* than its trained-longitude
counterpart. Rough implied trained-longitude corridor AUCs (an **estimate**, by
reweighting the h = 6 map; NOT a re-rolled measurement): e017 `s1_s0`
0.589 → ≈0.75, xl89 0.674 → ≈0.82, xl144 0.681 → ≈0.82.

**What IS qualified: the stencil advantage.** At h = 6, stencil head vs the
frozen 1-point gate:

| head · file | scope | total adv | in-block adv × share | out-of-block adv × share | **in-block share of the advantage** |
|---|---|---|---|---|---|
| xl144 s0 (#356) | corridor | +0.0907 | +0.182 × 0.239 = +0.0436 | +0.062 × 0.761 = +0.0471 | **48%** |
| xl144 s0 (#356) | window | +0.0798 | +0.187 × 0.250 = +0.0467 | +0.044 × 0.750 = +0.0331 | **59%** |
| xl144 s1 (#356) | corridor / window | +0.0865 / +0.0768 | | | **45% / 57%** |
| xl89 s0 (#355) | corridor / window | +0.0829 / +0.0723 | | | **43% / 57%** |
| xl89 s1 (#355) | corridor / window | +0.0849 / +0.0771 | | | **42% / 57%** |

**Roughly half of the stencil head's measured advantage is earned on pixels the
model was never trained on.** That is defensible — reading neighbours is exactly
how you cover a hole in your own coordinate input, and an untrained pixel's
neighbours include trained ones — but *"spatial coupling helps forecasting"* and
*"spatial coupling helps you extrapolate into a training hole"* are **different
claims**, and no published number separates them. Every corridor-AUC comparison
in `ml/LEADERBOARD.md` and the paper's Table 5 carries this qualification. The
ranking is unaffected (all arms are scored on the same pixels).

**The corridor paradox dissolves.** The previous reading — "the corridor is not
the skilful part of the window" (0.71 in vs 0.74 out) — was an artefact of the
same overlap. Split by the block: **out-of-block corridor 0.864 vs non-corridor
0.866 — indistinguishable**; in-block corridor 0.229 vs non-corridor 0.369. The
corridor is not selecting unskilful pixels. Fast-current pixels inside an
untrained longitude block are simply the hardest thing in the window to
extrapolate to, which is what one expects if the missing information is
advective.

**THE REPORTING DEFECT — found here, FIXED in the same session by the parallel
arm (`58aad7f`).** `rollout_spatial.py` knew about the block — it recomputes
`x_hold` at line 567 — and **recorded nothing about it in any artefact**. Every
`rollout_spatial.json` on `ml-metrics`, every corridor AUC in
`ml/LEADERBOARD.md`, and the paper up to this revision reported a single number
blended over trained and untrained longitudes with no field naming the split.
`58aad7f` adds a top-level `holdout_lon` block (bounds, column count, which
pools exclude it, per-scope `in_block`/`of`/`frac`) and a `*_trainlon` /
`*_holdlon` split beside every scope, plus `tests/test_roll_holdout_lon.py`.
**Every number already on `ml-metrics` predates that**, so the archive stays
blended and this entry is how those numbers are read. *(`rollout_spatial.py`
and `tests/` are the parallel arm's; nothing under them was touched here.)*

**DECISIVE CONFIRMATION STILL OUTSTANDING.** Everything above is consistent with
the diagnosis and no alternative survives, but the experiment that would settle
it beyond argument has not been run: **retrain one small stage-2 head with a
DIFFERENT held-out longitude block and confirm the band moves with it.** It is
not a one-line change — `temporal.py:1180` reads the block from the *frozen
codec's* saved args and has no switch of its own, so it needs a
`--train-lon-hold` knob on stage 2 (or a retrained codec). Cost once the knob
exists: minutes of GPU at pilot scale. Until then the diagnosis is
*overdetermined but unfalsified*, not *tested*.

**Where it landed.** `ml/paper/paper.tex` — the figure is redrawn with the block
hatched, outlined and named in the legend, plus a zonal-mean panel showing the
two cliffs; the caption and four body paragraphs are rewritten around the
finding; the glossary gains a **Holdout longitudes** entry; the abstract, §8.7
(dependency cone), §8.8 (standings) and Limitations (7) carry the
qualification. All contrasts reproduce on the arm's second seed to within 0.02
(`ml/CLAUDE.md` §3b — the block-centre *level* is the one exception, 0.171 vs
0.148, and is not quoted as a level).

---

<a id="seed-rule-2026-08-19"></a>
## OPERATIONS · The two-seed requirement is now conditional, 2026-08-19 ~11:40Z

Not an experiment. Recorded here because it changes how entries BELOW this line
may be written, and because future readers of a single-seed entry are owed the
reason it was allowed to be one.

Chris, 2026-08-19: *"I don't think we need to runs for every experiment. At
least when two experiments seem to agree a lot during our experience and the
confidence intervals seem small (let's quantify them using past data given exp
scale)."*

The quantification was done from this log, `ml/LEADERBOARD.md` and every
`probes-*.json` on `ml-metrics`, and the resulting standing rule is
[`ml/CLAUDE.md` §3b](https://blauewelt.github.io/earth/docs.html?f=ml/CLAUDE.md)
— "Replication is bought where variance lives, not everywhere". The table lives
there and **must be extended whenever a new replicate lands**; the rule has no
authority apart from it.

**The headline numbers.** Seed spread is a property of METRIC × SCALE, and this
programme spans two orders of magnitude of it:

| metric · scale | replicates | measured spread |
|---|---|---|
| rolled corridor AUC, xl tier (205–217M, 60k–200k) | 5 pairs + 1 triple | pair \|Δ\| 0.0020–0.0051; pooled sd **0.0021** (7 dof), 95% upper bound 0.0037 |
| rolled corridor AUC, 88M tier | 4 triples + 2 pairs | ranges 0.0011–0.0150; pooled sd **0.0056** |
| rolled corridor AUC, 34M tier | 14 configurations | ranges to 0.0224; pooled sd **0.0070** |
| transport band r (`amoc_bands`), the SAME xl checkpoints | 5 pairs + 1 triple | per-band spreads **0.003–0.119**; pooled sd **0.041** (15 dof) — **20×** the corridor's, same files. Re-mined 2026-08-20 on #418; the old "0.05–0.07" was two groups summarised to one number each and is RETIRED |
| rolled corridor AUC, `_trainlon` scope, xl tier | 2 pairs (#417, #418) | \|Δ\| 0.00075, 0.00108; pooled sd **0.00066** (2 dof) — TIGHTER than the blended scope |
| rolled corridor AUC, `_holdlon` scope, xl tier | 2 pairs (#417, #418) | \|Δ\| 0.00958, 0.01933; pooled sd **0.01079** (2 dof) — **16.4×** `_trainlon`. **Never quotable as a level** |
| RAPID head k-fold, 1.8M · 6k (E-010) | 2 triples | range **0.245**, sd 0.123 |
| RAPID head k-fold, 1.8–10.7M · 60k | 5 triples | pooled sd **0.095** |
| codec head probe, 0.92M · 40k · 1° | 1 codec-seed pair | head **0.036**, ridge 0.012 |
| anything at pentad or daily cadence | **none** | **UNMEASURED** |

Corridor AUCs are recomputed to five decimals from each head's twelve archived
per-horizon `msss_clim` values, since the stored `horizon_auc` is rounded to
three.

**What it permits.** A result scored by rolled corridor AUC, at ≥205M / ≥60k
steps on the frozen f3 anchor codec and the monthly `family3_na025` tensor, may
stand on **one seed** if its claimed effect is **≥ 0.025** — five times the
largest pair delta ever measured at that tier (0.0051, E-032 xl144), and also
five times the sd of a single-seed difference at the 95% upper bound of the
tier sd. Two seeds remain mandatory for any probe-scored claim, any new metric
/ cadence / tensor / codec / scale tier with no measured pair, any paper
headline, and any claim that an effect is zero or that an axis is closed.

**Two things that are not replication and must not be logged as spread.**
`e017_u1_s0` has reproduced gate AUC 0.643 / corridor 0.589 / window 0.622 in
**eighteen** eval runs (#228 … #413), and `probe_kfold` on the f3_anchor41M
codec over the pentad tensor returned rapid r 0.660 identically in #390, #392,
#397 and #406 — that is PROTOCOL determinism, the certificate that makes an
eval wave readable, not a seed measurement. Separately, the 0.041 head-k-fold
box effect (E-008) is an environment term, not a seed term.

**Why the rule is worth having.** A second seed at the xl tier is ~15.6 h and
~$4.6 of training (#396) plus ~3–5 h and ~$1.0–1.5 of eval (#413) — about $6
and a day of a rented box — against a metric that reproduces to 0.002–0.005.
At probe scale the same replicate is not optional: the 0.245 measured in E-010
is what killed E-005's +0.28 unroll result, which had stood on two n = 1 runs
from different dispatches scored on a 36-month single split.

`ml/plans/E033_scale_program.md` §7b carries the consequence for the planned
waves; `docs/ML_BASICS.md` §4 points at the rule from the protocol section.

---

<a id="credit-triage-2026-08-19"></a>
## OPERATIONS · Credit triage, 2026-08-19 01:35Z — two runs cancelled so three could finish

Not an experiment. Recorded here because it changes what several entries below
are waiting for, and because the arithmetic is the whole argument.

**The arithmetic, read at 01:25Z.** Credit **$6.21**, burn **$1.784/h** (five
boxes plus storage), exhaustion **~04:53Z**. That deadline lands mid-flight for
EVERY run in the fleet — including **#396** (the E-035 seed-0 re-run), which by then would be at ~97% of
its 200,000 stage-2 steps. An exhaustion that stops five boxes at once does not
produce five partial results; it produces five zero results, and the largest of
them would be a 200k run lost in its final hour.

**Two runs could not finish on any remaining credit, under any allocation.**
**#400** (E-038c, the daily codec) had **~28 h** to go — its probe cadence alone
is **65% of wall clock** at daily scale. **#408** (the E-042 SST arm) had
**~14 h** to go. No triage makes either fit inside $6.21.

**DECISION (main session): cancel #400 and #408, and stop their boxes** (Vast
**47913006** and **47724565**). They are the right two to cut precisely because
both are cheap to RESUME: #400's box keeps the **165.6 GB daily tensor** plus a
**~22k-step orphan checkpoint** that the rescue step will pick up, and #408's
box keeps whatever of the r2 build it completed. Neither cancellation destroys
work that would have to be re-earned from nothing.

**After triage:** burn **$1.184/h**, credit **$5.86**, runway **~4.9 h**. Inside
that runway, **#401** (the E-036 eval, ETA 03:12Z), **#409** (E-038a's own
codec through the head probe, ~03:30Z) and **#396** (~06:06Z) all finish — and the hourly fleet-health check stops each box as it goes idle,
so the runway lengthens as the runs land rather than being spent on idle GPUs.

**Re-dispatch #400 and #408 VERBATIM the moment credit is topped up.** Their
inputs survive in provenance — the same mechanism that recovered the inputs of
#389 (E-038c, the first daily codec) verbatim for #400 — so neither needs reconstructing, which is the expensive
failure mode (#393, the E-036 eval, §"#393 died with nothing archived").

**Also DEFERRED by this: the stage-2 comparison matrix** — arm A `!run-386`
against arm B `!f3_anchor41M`, temporal 384×6 @ 24k steps, `head_probe` on
both. It waits on **(i)** credit and **(ii)** #409's head-vs-raw verdict. (ii)
is the real gate, not the money: if #409's head does not beat its own raw
control, the fresh-codec arm is not obviously the right next spend and the
matrix would be measuring the wrong axis.

### UPDATE, 2026-08-19 ~04:00Z — credit restored, and the triage's three runs all landed

**Balance back to $53.40** on a top-up. The runway argument above is spent and
the decisions it forced are being unwound in order: **#400** (E-038c, the daily
codec) and **#408** (the E-042 SST arm) are being **re-dispatched VERBATIM** from
their archived provenance — a separate step, not this one — and both boxes still
hold what makes a re-dispatch cheap (#400's 165.6 GB daily tensor and its ~22k
orphan checkpoint, #408's partial r2 build).

All three runs the triage was built to protect finished inside it: **#401**
(E-036 eval, gate passed, §"E-036 RESOLVED" below), **#409** (E-038a verdict,
§"VERDICT, 2026-08-19 ~03:30Z" below), and **#396** (the E-035 seed-0 re-run), which is **on pace at
170,000 / 200,000 steps, ETA ~06:10Z**. The triage cost two runs and saved
three, which is the trade it was written to make.

**One decision does NOT unwind with the money.** The stage-2 comparison matrix
above was deferred on two gates, credit and #409's verdict. Credit is no longer
a gate; **#409's verdict is now a reason of its own** — its head did not beat its
own raw control, so both arms of that matrix are representations just measured
as equivalent to raw pixels. It stays down-prioritised on evidence and should be
re-scoped around a raw arm before it is re-dispatched (E-038a VERDICT §(d)).

**AMENDED 2026-08-19 ~11:00Z — this de-prioritisation is REVERSED.** The matrix
is re-prioritised: probe parity on the CURRENT state is not evidence about
forecast-substrate quality, and "which embedding rolls forward better" is the
programme's actual question. See E-038a VERDICT §(e).

**What #400 bought before it was cancelled**, recorded so that 22k steps are not
a total loss — these are the first daily-scale numbers this programme has:
**22,000 / 200,000 steps**, **216.6 ms/step** pure training, **probes 65% of
wall clock** at **~2,300 s per full eval** (which is why the daily arm costs 28 h
and not 12), and a light-probe `linear_r_deseas` of **0.626 @ 7.5k → 0.554 @
15k → 0.581 @ 20k** (the last of those a light rung). Against the pentad
wind-only bar of **0.670** this is **not yet a comparison**: the **daily wind bar
is unknown** — the probe ladder never reached it — and these are single-split
light probes (rule 5), not `probe_kfold`. The non-monotonicity across
7.5k/15k/20k is what a light probe does at this noise level and is not a trend.

### RESURRECTED, 2026-08-19 04:04Z — #400 → **#410** (E-038c daily codec), #408 → **#411** (E-042 SST arm)

Both boxes came up on the first `gpu_box.mjs start` — no `resources_unavailable`,
and neither sits on the two hosts that refused yesterday (these are machines
**137260** and **137510**, not 145738/70981). Both runners were `online` and
**idle** before either dispatch, and `47718230` (#396, the E-035 seed-0 re-run, ETA ~06:10Z) was not
touched. Exactly two dispatches.

| | #410 (was #400) | #411 (was #408) |
|---|---|---|
| box | `gpu-box-46996216` · Vast **47913006** · 700 GB | `gpu-box-39184683` · Vast **47724565** · 504 GiB RAM |
| `head_sha` | `cac9a017` | `6bc9ef7b` |
| rescue step | **85 s** (was 1 s in #400) | **1,400 s** (was 1,490 s in #408) |
| build step | **1 s** — tensor warm | **0 s** — the r2 build is warm |
| provenance | 351 s (re-hashing 165.6 GB) | 244 s |
| `Seed resume checkpoint` | **skipped** | **skipped** |

The two runs took **different shas** because `main` moved between the
dispatches (`6bc9ef7` landed 40 s after #410 went out). The delta is
`EXPERIMENTS.md` and the paper only — no code, no workflow — so the two runs
are code-identical; both are current `main` as of their own dispatch instant.

**#410 is #400 verbatim.** All 24 non-`doc` fields were copied from
`probes-389.json`'s `provenance.json.inputs` — the archived artefact, not the
plan — and the list in the hand-off matches it field for field: 200,000 × 512,
codec 512×12, 4 heads, `d_dec` 256, `d_z` 32, `patch` 1, `anomaly`,
`family5_na025_daily`, `eval_every` 7500, `light_probe_every` 10,000,
`temporal_steps` 0 (`temporal_d_model` 96, `temporal_layers` 3),
`head_probe` false, `window` global, `sst_channel` false, `resume` empty,
`max_minutes` 2200, `job_timeout` 2600, `lr_floor` 0, `lr_decay_steps` 0.

**The warm boxes did what they were kept for, and the STEP DURATIONS say so
rather than the step names.** `Build dataset` **1 s** on #410 and **0 s** on
#411 — against the **1,921 s** #408 paid to build the r2 tensor cold. Both
short-circuited; nothing was rebuilt.

**The orphan rescue RAN — and a rescue is not a resume.** Step 2 took **85 s**
on #410 (against **1 s** in #400 itself, which had no orphan to find) and
**1,400 s** on #411, and the effect is checkable OFF the box rather than from
the step's own colour: `rescued-orphan-latest-410.pt` (456 MB, 04:05:31Z) and
`rescued-orphan-latest-411.pt` (456 MB, 04:13:16Z) are now assets on the
`model-checkpoints-v1` release, each with its `rescued-orphan-metrics-latest-*`
sidecar. Both cancelled runs' weights are durable off-box for the first time.

But the step **preserves** an orphan; it does not seed one. `--resume
orphan-latest` is what reads it (`ml-train.yml:133`), the `Seed resume
checkpoint from the release` step is `if: always() && inputs.resume != ''`
(`:763`), and the trainer is invoked `--resume "${RECIPE_RESUME:-inputs.resume}"`
(`:868`) — so with the verbatim empty `resume`, that step **skipped on both
runs** and **both codecs train from step 0**. That is the documented behaviour,
not a regression, and it was not fought: #400's 22,000 daily steps are now
durable, they are simply not continued. Continuing them would need
`resume: "orphan-latest"` — a different dispatch, and one whose architecture
match nothing has checked.

**#411 is #408 verbatim plus `head_probe: "true"`** — eval-side only, so the
arm stays weight-comparable with #386 (E-038a, the f4-40M control on r1) — on
`window: recipe:f4r2-40M`, which PINS `family4_na025_pentad_r2` and beats
`inputs.tensor` (the lesson of #405, this arm's first, cancelled dispatch).
**One field could not be recovered verbatim and is stated as such:** #408's
`job_timeout`. Its logs are gone (GitHub returns 404 for a cancelled run's log
blob) and no `provenance.json` was ever archived for it, so the two surviving
records disagree — the prepared 25-field block above says **1500**, while the
2026-08-18c hand-off and #408's own run name both quote **1000** for its
predecessor #405. **1500 was used**, because `job_timeout` is a timeout and
touches no number the run produces, while 1000 min = 16.7 h against a warm run
of ~16 h would put an artificial death inside the error bar of the estimate.
`max_minutes` stays **0**, as recorded.

**Both verified IN Train, from the artefact rather than the step colour** (§2).
Each run's own `config` line on `ml-live-<n>` reproduces its predecessor's field
for field — #410: `family5_na025_daily.npz`, C **39**, T **15,706**, 37.976 M
params, `resume` **null**, `recipe` null; #411: `family4_na025_pentad_r2.npz`,
C **40** (the SST channel is present; r1 is C=39), T **3,142**, 37.976 M params,
`recipe` **f4r2-40M**. First numbers, against the runs they resurrect:

| | #410 step-0 | #400 step-0 | #411 step-0 | #408 step-0 |
|---|---|---|---|---|
| `linear_r_deseas` | 0.556 | 0.526 | 0.518 | 0.588 |
| `linear_r_raw` | 0.582 | 0.548 | 0.533 | 0.576 |
| `probe_seconds` | 2,281.8 | 2,310.0 | 870.5 | 1,282.9 |

Those are RANDOM-INIT probes and differ only by initialisation and batch draw;
they are a liveness check, not a result. The line that does carry information is
`chan_mse_persistence` = **0.44384765625** on #410, **bit-identical** to #400's —
a data-only quantity no model can move, i.e. the same tensor, on the same box,
to the last float32 digit. First loss lines are on trajectory (#410 step 2,000
`loss_rec` 0.26655 / `loss_nei` 0.20238 against #400's 0.28569 / 0.20491 at step
1,000; #411 step 11,000 `loss_rec` 0.23396 at ~155–198 ms/step, #386's measured
pentad pace being 190). Both on GPU — 93% at 72 °C and 100% at 71 °C — which is
the one check that catches the wrong-device failure. **And #410's step counter
starts at 0 with `resume: null`: the rescue preserved the 22k checkpoint and did
not continue it, exactly as the code above says.**

---

<a id="e-042"></a>
## E-042 · SST as channel 40 — DISPATCHED 2026-08-18 21:55Z as #405

Chris, 2026-08-18, reading the channel table: the embedding *"should
represent a holistic view on any point of the world"*. The 39 channels are
almost entirely AMOC plumbing, and the tensor's only temperature is Argo
`rg_t`.

Full reasoning: [the E-042 plan](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E042_sst_channel.md).

**A numbering note.** Four commit messages from 2026-08-18 (`bcacf89`,
`e2438c7` and the two follow-ups) and the code comments they landed label this
work **E-041**, which was already spent on the globe playback feature that
shipped the same day. The work is **E-042**; the code and this log are
corrected, the commit messages are not — rewriting published history for a
label is the worse trade. Anyone grepping `E-041` in the SST code is in the
right place.

### The gap, in coverage percentages

`build_family4.fill_rg_pentad` walks the RG cubes from
`y, m = 2004 + k // 12, k % 12 + 1` — **Argo starts in 2004**. So 1982–2003,
**22 of the 43 years on the axis, carry no temperature at all**, and inside
the Argo era E-034 §4's one-live-bin-per-month policy leaves `rg_t` a missing
token in **83.6%** of pentad bins and **96.7%** of daily bins (**92%** and
**98%** over the whole axis). OISST v2.1 is daily, native $0.25^\circ$ — the
tensor's own grid, not an upsample — and live in ~100% of bins across the
whole 1982–2024 axis.

### Hypothesis, control, and what would falsify it

**Hypothesis.** A codec of the same capacity trained on the 40-channel r2
pentad tensor reads the RAPID transport better than the same codec trained on
r1, because 22 of 43 years gain their first temperature field and the added
channel is the only one native to the tensor's grid.

**Control: run #386** (E-038a, f4-40M on r1) — a matched A/B in which
`tensor: family4_na025_pentad_r2` is the ONLY difference from a dispatch
otherwise copied verbatim from #386's own `INPUTS_JSON`. **The frozen anchor
is NOT available as a control here**: `f3_anchor41M` has a 39-row `chan_emb`
and cannot encode a 40-channel tensor at all, so E-038's frozen-codec baseline
retires on r2 and the only surviving external baseline is the wind-only ridge,
**0.670** at pentad.

**Falsified** if the r2 arm's `probe_kfold` rapid r does not exceed #386's, or
exceeds it by less than the seed band (sd 0.123, E-010), at matched steps. The
discriminating follow-up is the **1982–92 block**, where GLORYS is absent: if
r2 wins only there, SST is a coverage fix rather than an information gain.
**A result that would retire the channel**: r2 measurably worse than r1 beyond
the seed band — the fortieth channel costing capacity it does not repay.

### The four scale numbers (rule 6)

| | #386 (control, r1) | E-042 arm (r2) |
|---|---|---|
| codec params | 37,975,889 | **37,976,465** (+576: one row each in `chan_emb` and `q_chan`, measured by instantiating the real class at `n_chan=40`) |
| batch | 512 | 512 |
| steps | 200,000 | 200,000 |
| data points | train pool 191,520,806 pixel-pentads | the same pool; the OBSERVED-value count rises by the sst channel and must be **read from the build's own Chinchilla inventory**, not scaled by hand |

Everything else is held: `d_z` 32, `patch` 1, 512 × 12 × 4 heads, `d_dec` 256,
`anomaly` true, `eval_every` 7,500, `light_probe_every` 10,000, `resume` null.
`head_probe: "true"`, because at pentad cadence the unpooled head is the
primary read-out (§3).

### What exists, verified

- `ml/fetch_sst_na.py` — OISST v2.1 streamed one year at a time from PSL,
  cropped to the E-034 window and **bilinearly** interpolated onto the
  tensor's axes. Bilinear is load-bearing: OISST's lat/lon are cell CENTRES at
  $0.125 + k\cdot0.25$ while the window samples ON multiples of $0.25$, so
  every target falls exactly halfway between two source centres in each axis
  and nearest-indexing would displace the whole field by half a cell,
  invisibly. Test 2 measures an analytic ramp's reproduction at
  **7.6e-07 °C**. Measured on 1993: 365 rows in **14.5 s**, 0.26 GB peak RSS.
- `.github/workflows/sst-na-bake.yml` — **run #1 completed**, and the bytes
  are verified on the Hub (`chfrank/earth-tensors`): `sst_na025/index.npz`
  **146,802 B**, `sst_na025/sst_daily_na.npy` **4,245,677,460 B**. That second
  number is the artefact checking itself — $15{,}706\times281\times481\times2
  = 4{,}245{,}677{,}332$ plus numpy's 128-byte header — so the file is the
  whole axis, not a run that stopped early.
- Recipes **f4r2 / f5r2**: `CHANS_R2 = list(f3.CHANS) + ["sst"]`, appended so
  channels 1–39 keep the indices every published result was measured at (a
  test pins r2's channels 0–38 bit-identical to r1 from the same fixtures),
  each `(cadence, rev)` with its own output name so an r1 build in flight is
  never overwritten. Two new VALUES on the `tensor` input, never a 26th input.
  `tests/test_e034_family4.py` 14/14 and `tests/test_e034_family5.py` 8/8.
- The climatology is deliberately **not** shared with the app's SST bake: this
  emits the FIELD; the pipeline's baseline is train-years-only, the app's is
  the WMO 1991–2020 normal which includes the holdout years (E-040 §5).

**Blocked on disk, not on a decision.** The r2 pentad tensor is
$[3142,281,481,40]$ float16 = **34.0 GB** and the build wants ~42 GB free
while the memmap and the archive coexist; no box has that headroom while
#386/#387 (the E-038a/b pentad codecs) hold the 126 GB boxes on r1.

### Dispatch attempted 2026-08-18 ~20:20–21:00Z — NOT DISPATCHED, and the blocker MOVED

The disk arithmetic now clears, and it was checked against the artefacts rather
than re-derived. `gpu-box-47566395` (Vast **47718224**, 100 GB) reports 46 GB
used → **54 GB free**, against a peak of:

| item | bytes | source of the number |
|---|---|---|
| pentad025 base fields | 5.058 GB | `Content-Length` of the four `pentad_mean_*.npy` on the Hub, 1,264,566,444 B each |
| SST artifact | 4.246 GB | `sst_na025/sst_daily_na.npy` 4,245,677,460 B + its index |
| dense `_build.npy` memmap | 33.974 GB | $3142\times281\times481\times40\times2$ exactly |
| compressed archive, coexisting with the memmap | ~4.2–8 GB | estimated, **not measured** — §7 above guessed the pair at ~42 GB |
| **peak** | **47.4–51.3 GB** | **~2.7–6.6 GB of margin** |

The builder's guard is `need > free * 0.95` on the memmap's own directory, i.e.
it demands **≥ 35.76 GB free** at the moment it runs — comfortably met — and
since `d3ea240` it sits AFTER the recipe short-circuit, so it can no longer
refuse a build that was about to be skipped.

**What stopped it was the box, not the bytes.** `47718224` would not start:
sixteen start calls spread over forty minutes, 20:20Z to 21:00Z, every one of
them `resources_unavailable` ("state change queued"), and the instance's own
`intended_status` / `next_state` never left `stopped` — so the queued change is
not pending, it was refused. Its host (machine 145738) is full, which also rules
out its sibling `gpu-box-47566393` (47720655, same host, and 46 GB free anyway).
No other box in the fleet clears the precondition: `gpu-box-39184683` (47724565,
57 GB free, 515 GB RAM) is the one that would, and it refused to start the same
way; `gpu-box-47529389` has 30 GB free, `gpu-box-47094145` 37 GB,
`gpu-box-30257785` 45 GB, `gpu-box-42005419` 25 GB. The four boxes that were
busy stayed untouched, and no box was rented — a precondition is not something
to lower until a dispatch fits through it.

**The dispatch is prepared, and its 25 fields are recorded here so the next
session copies rather than reconstructs.** Every training-relevant field was
cross-checked against **#386's own `config` line on `ml-live-386`** — the
E-038a control arm's artefact, not the plan — and matches exactly:

```json
{"doc": "E-042 SST A/B: the FIRST r2 codec …",
 "steps": "200000", "batch": "512", "d_z": "32", "anomaly": "true",
 "temporal_steps": "0", "temporal_d_model": "96", "temporal_layers": "3",
 "eval_every": "7500", "resume": "", "head_probe": "true",
 "light_probe_every": "10000", "window": "global",
 "tensor": "family4_na025_pentad_r2", "sst_channel": "false", "patch": "1",
 "max_minutes": "0", "runner": "gpu-box-47566395", "job_timeout": "1500",
 "lr_floor": "0", "lr_decay_steps": "0", "codec_d_model": "512",
 "codec_layers": "12", "codec_heads": "4", "codec_d_dec": "256"}
```

`resume` is empty **by construction, not by omission**: the 39-row `chan_emb`
of `f3_anchor41M` cannot encode 40 channels, so there is nothing to seed from
(ml/CLAUDE.md §1 — an omitted input is the DEFAULT, never an inheritance).
`head_probe` is the one field that is NOT copied from #386, and it is eval-side
only: probes do not touch trained weights, so the arm stays weight-comparable
while gaining the unpooled read-out §3 makes primary at this cadence.

**One confound to state now rather than discover later.** #386 built its own r1
tensor on `gpu-box-47094143`, and that box is occupied until #386 finishes, so
the r2 arm will build its own tensor on a DIFFERENT box whichever box it lands
on. Family 4 has no pinned pull from `data-cache-v1` the way family 3 does, so
the A/B is cross-box either way and the box-effect measured at family 3
(0.041 on the head k-fold) is not excluded by construction. That is a property
of the design, not of which box tonight's dispatch would have landed on — but
it is worth pinning the r2 tensor to `data-cache-v1` the way family 3 is
pinned, before the comparison is quoted.

**Cost so far:** one `ubuntu-latest` bake run, no GPU. Tonight added no GPU
cost — the box never started, so nothing was billed beyond storage.

### Dispatched — #405 (the E-042 SST arm, first attempt), 2026-08-18 21:55Z, `head_sha` 78d66a6

**The blocker above cleared within the hour, and it cleared on a box nobody
had costed.** The earlier survey read Vast's `disk_usage` as a percentage; it
is in GB, and `cpu_ram` is in MiB. Re-read in absolute units, one box in the
fleet cleared both bars at once.

Box **gpu-box-39184683** (Vast 47724565): 57 GB free of 100 against the ~42 GB
the r2 build needs, and 504 GiB RAM against the 126 GB class family-4 requires
(#368 host-OOMed on a 63 GB box). It was also the fleet's one idle-burning box,
so the arm and the waste cancelled. Inputs are the E-038a control #386's 24-field `INPUTS_JSON`
verbatim, plus `resume: ""`, with `head_probe` `false → true` and `window`
`global → recipe:f4r2-40M`.

**The plan's own dispatch instruction was wrong, and would have produced a
green run reporting SST numbers for an experiment with no SST in it.** §5 said
to dispatch `window: recipe:f4-40M` with the tensor overridden.
`ml/recipes/f4-40M.json` PINS `"tensor": "family4_na025_pentad"`, and every
consumption site — `ml-train.yml:762`, `:881`, `scripts/probes_run.sh:43`, and
the build branches at 378–412 and 518–591 — reads `${RECIPE_TENSOR:-$IN_TENSOR}`.
The `:-` fallback fires only when `RECIPE_TENSOR` is UNSET, so the recipe wins
and `inputs.tensor` is discarded entirely, while `provenance.json` (raw
`toJSON(inputs)`) would have faithfully recorded `family4_na025_pentad_r2`.
Intent and reality disagreeing in exactly the way the manifest exists to catch.
A recipe's tensor is **not overridable from a dispatch**, so the r2 arm got its
own recipe, `ml/recipes/f4r2-40M.json`, and the plan is corrected. Note the
`resolve_recipe.sh` header comment claiming "a recipe cannot silently override
something a dispatch stated on purpose" is false for any key the recipe names.

**Result: PARKED, not pending.** The arm's live run at the time of writing was
**#408** (this arm's second dispatch) on Vast **47724565**, and it was **cancelled 2026-08-19 01:35Z with
~14 h still to go** in the credit triage at the top of this file — a money
decision, not a scientific one. Its box keeps whatever of the r2 build it
completed, so the arm resumes warm: **re-dispatch #408's inputs verbatim once
credit is topped up.** — **DONE 2026-08-19 04:04Z as #411** (this arm, resurrected), on the same box,
inputs verbatim plus `head_probe: "true"`; its `Build dataset` step took **0 s**,
so the warm r2 build was real (§"RESURRECTED" above). When it does land it must
be read on the #406 protocol (E-038's read-out ladder, attempt 3)
— head against its own matched raw control, not head against wind (see the
E-038 read-out resolution below).

---

<a id="e-038"></a>
## E-038a/b · The first codecs trained ON the pentad tensor — DISPATCHED 2026-08-17 ~18:20Z

Chris, 2026-08-17: *"let's change the plan to retrain the codec (40m and 200m?)
on the new data (1 day, 5 days) … the reason is that we have a new kind of
data that is out of domain for the existing codec"*, and then *"Please start
(also the 200M rung) both on pentad and daily."* These two arms are the pentad
half. The daily half (E-038c/d) needs a tensor that does not exist yet.

Full reasoning: [the E-038 plan](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E038_codec_matrix.md).

### STATUS 2026-08-18 22:30Z — the codec trained cleanly; NEITHER number exists

Both headline numbers this arm and its read-out ladder were dispatched to
produce were lost to instrumentation, not to science. Recorded here at the
same weight as a result, per rule 4, because the pattern is the finding.

**#386 (E-038a f4-40M, `head_sha` c7ba151) — trained clean, probe ladder
annihilated.** Stage 1 ran 06:47:53 → 20:58:34Z and finished at step
**166,752**: `fit_schedule` correctly re-fit 200,000 down to fit the 850-minute
budget, so the cosine annealed to zero rather than being cut off (this is the
mechanism that was broken on the VOID #366; it worked). Loss finite throughout,
`loss_rec` 0.297 → 0.220, **190 ms/step steady** — matching the recorded rate
for this configuration exactly, i.e. no lemon-box signature. The in-training
light probe (single 36-month split, noisy — rule 5) rose `linear_r_deseas`
0.564 → **0.624** and `temporal_r_deseas` 0.531 → 0.604, with
`chan_vs_persistence_pct` 4.0 → 31.8.

Then **all three probes were SIGKILLed by the host OOM killer**, at 21:10:29
(`probe_sequence`), 21:38:23 (`probe_kfold`) and 22:03:56 (`dip_check`) — each
~28 min in, immediately after the anomaly transform, on the full-size
`np.isfinite(Xa)` bool: **16.56 GB** of bool live alongside the 33.1 GB float16
tensor. Each rung was individually `|| echo "::warning::…"`, so all three
warnings fired and the run reported **success** with an archive containing only
`provenance.json`. This is NOT the #131 `bash -e` shape — nothing aborted the
step; every rung ran and every rung died.

**The fix was already on `main` while the run was still training.** `70ffe2d`
("The probes get the LazyPixels treatment") landed 08:39:15Z — 2 h 33 m after
#386 checked out c7ba151, and **12 h 21 m before** #386 reached its probes. A
long run's code is frozen at checkout, so a fix that lands mid-run does not
reach that run's probe phase. #388 (the frozen control) hit this identical defect
and was re-dispatched as #390, which succeeded; #386 was left to walk into it.
**A 14-hour run and a 30-minute eval should not share a checkout.**

What survives: `eval.json` is NOT empty — per-channel reconstruction skill
(best `rg_t400` 0.941, `rg_t500` 0.929, `rg_t300` 0.896; worst `cur_speed`
0.022, `ssh` 0.052) and a `t+1` result that **beats persistence** (mse 0.958 vs
1.317). Only `rapid_probe` is NaN — and note it was WRITTEN as NaN, which
`docs/INFRASTRUCTURE.md` §4 invariant 12 forbids ("a result file is never
written containing NaN — the job stops instead"). That guard did not fire.

The trained codec is now durable: verified as the real artefact (`tag`
`run-386`, `step` 166752, 37,975,889 params, `args` matching the dispatch —
NOT the `rescued-orphan-latest-386.pt` on the release, which is an earlier
job's leftover carrying #386's run number) and uploaded as
**`run-386__pixelmae.pt`** (455,908,861 B) and `run-386__metrics.jsonl` to
`model-checkpoints-v1`, re-downloaded and hashed to confirm. Before that it
existed only in an Actions artifact expiring 2026-09-17 and on one rented
disk — invariant 7 ("recoverable from the release without a GPU") was violated
for a 14-hour result. **The re-score is the next action**: eval-only at current
`main`, warm tensor, ~30 min.

**#392 / #397 (the read-out ladder) — the unpooled head still does not exist.**
The question is whether E-038's headline is read-out-limited or
representation-limited: `probe_kfold`'s ridge reads `Z.mean(1)` over the 26.5°N
section, and geostrophic transport IS the east-minus-west contrast across that
line, which a mean annihilates by construction. Two attempts, two different
deaths, both green:

- **#392** — pooled ridge landed (rapid **0.660** [0.593, 0.722], n 1459, RMSE
  2.97 Sv, against a wind-only bar of **0.670** [0.601, 0.733]: two raw wind
  features still beat 64 mean-pooled learned ones). Both `probe_head.py` calls
  OOM-killed by an 82.8 GB transient inside `np.nan_to_num(Xa, copy=False)` —
  `copy=False` never copies the values, but the masked-copyto form builds
  `isnan` + `isposinf` + `isneginf`, and the latter two each build `isinf` and
  `signbit` underneath: five full-size bools live at once, 132.5 GB peak on a
  126 GB box. Fixed in `f2ee8b8` (LazyPixels).
- **#397** — the fix WORKED. Both invocations cleared the full anomaly
  transform and the entire 3,142-month embedding pass with no OOM marker
  anywhere. They then died on the **first `loss.backward()`** in `fold_fit`
  with `RuntimeError: Failed to find C compiler. Please specify via CC
  environment variable or set triton.knobs.build.impl` — `SectionHead`'s
  cross-attention backward dispatches to a Triton-JIT kernel, Triton builds its
  CUDA-utils C extension on first use, and the box has no compiler and no `CC`.
  Cost: 24.5 min of GPU, the anomaly transform run twice, for an error
  decidable in the first second.

  And behind it a second bug that had never fired because the first masked it:
  `fold_fit` ended `net(Xte).numpy()` with no `.cpu()`, which raises on CUDA.
  **The GPU read-out path had never once executed end to end**, though the
  comment above it asserted the move was deliberate and done — §0.1, verify the
  artefact, not the intention.

  Both fixed in **78d66a6**: `.cpu().numpy()`, plus `_usable_device()`, which
  runs one real `SectionHead` forward+backward+`opt.step` on the preferred
  device **before** the checkpoint, the anomaly transform and the embedding,
  and falls back to CPU on any exception with the reason printed. §0.3/§5.16 —
  a precondition that depends only on the inputs must be checked while the
  inputs are all it has cost you. `torch.cuda.is_available()` was TRUE on the
  box that failed, so the self-test exercises the same forward and backward
  that died. The global RNG is saved and restored around it, so the fold
  numbers stay a function of the data and the seed and never of the device.
  `--head-device auto|cpu|cuda` makes it reversible from a dispatch.
  `tests/test_head_device.py` pins all of it.

**Standing lesson from the three runs together.** Every one of them was GREEN.
The archive's file list was the truth each time and the run's colour was not —
exactly the failure signature §7 already names. Three consecutive experiments
lost their headline number to a best-effort guard reporting success, and in
each case the run cost hours and the diagnosis cost minutes. Best-effort is a
promise about *delivery*, never about *reporting*.

### Hypothesis, and what would falsify it

Every run for six weeks froze the codec (`resume: "!run-62,run-63"`). Applied
to pentad fields that would embed **out-of-domain data with an in-domain
codec**. The shift is not a vibe — on 32 of 39 channels the RG `missing`-token
share goes from ~0% at monthly to **~83% at pentad**, because E-034 §4 puts one
live RG timestep per month at every cadence. A codec that has essentially never
seen the `missing` token is being asked to spend most of its capacity on it.

**Hypothesis.** A codec of the SAME capacity retrained on the pentad tensor
beats the frozen monthly codec applied to that tensor.

**Control**, and it is the point of the design: `f3_anchor41M` (run #62,
40.7 M) evaluated on the family-4 tensor by `probe_kfold.py`. Capacity is held
fixed at ~40 M and only the data changes, so the difference IS the domain
shift. **Falsified** if the fresh 38 M codec does not beat that baseline — in
which case the reordering was unnecessary and the cadence work should go back
to reusing the codec. That control is an evaluation pass with no training loop;
it is the cheapest number in the plan and the only one that can falsify the
premise, so it is not optional.

The 200 M rung asks the independent question — does the answer depend on
capacity? — and wave 6A says it might: znoise-big55 at 88 M beat xl55 at 205 M
on the rolled corridor AUC, so "scale pays most" is already false on one axis.

### The four scale numbers (rule 6), MEASURED at build time on #365

| | f4-40M (E-038a) | f4-200M (E-038b) |
|---|---|---|
| codec params | **37,975,889** (512 × 12, d_dec 256) | **201,962,577** (1024 × 16, d_dec 512) |
| batch | 512 | 512 |
| steps | 200,000 (cosine, re-fit by `max_minutes`) | 200,000 |
| data points | train pool **191,520,806** pixel-pentads of 272.4 M (holdouts removed: 219/3142 months, 80/481 lon cols); 86,698 ocean cells of 135,161 | same tensor |

`build_family4.py` measured the Chinchilla inventory itself:
**2,252,509,289 observed values → a 112.6 M anchor**, against E-038 §2b's
prediction of 2,204.0 M → 110.2 M. **2.2% out**, fully explained by 86,698
ocean cells where the prediction assumed family 3's 84,405. The rungs bracket
it at 0.34× and 1.79× and stand as chosen.

**Why 200,000 steps for both, and what that costs the 200 M rung.** At batch
512 the train pool carries ~8.27 observed values per pixel, so 200 k steps is
**~848 M observed values** — 1.1× the Chinchilla optimum for 38 M params, and
within 3% of the 829 M that run #62 itself saw at monthly cadence. So E-038a
is matched to its control in capacity *and* in values seen, which is as clean
as this comparison gets. The same budget is **0.21× the optimum for 202 M
params**, deliberately: budget-matched is the comparison that isolates
capacity. One epoch of the pentad pool is 374 k steps and a compute-optimal
200 M run wants ~4.0 G values, i.e. 2.5 epochs — that is the natural follow-up
if the 200 M rung looks starved rather than saturated.

### Three defects sat between run #365 and a trained codec

#365 (04:07Z) was killed by the **host** OOM killer, not the GPU, and it died
early enough that everything downstream stayed unexercised. Fixed and pinned
before this dispatch, each by a test that fails on a laptop in seconds rather
than on a rented 4090 in hours:

1. **Host residency.** `LazyPixels` removed 49.7 GB of resident copies;
   `obs_any_chunked` + `pool_idx` remove the actual PEAK — a full `[T,H,W,C]`
   bool plus a `[T,H,W]` int64, live together, 18.8 GiB here and 94 GiB at
   daily. Measured 3.7× lower VmHWM, values and ORDER pinned against the
   originals (`tests/test_train_pool_memory.py`).
2. **float16 into float32 weights.** Family 4 is the project's first float16
   tensor: `mat1 and mat2 must have the same dtype, but got Half and Float`, on
   the first forward pass.
3. **float16 as the loss TARGET.** With (2) fixed, the backward pass fails
   instead: `Found dtype Half but expected Float`.

Nothing else in the chain is unexercised now: `tests/test_float16_training.py`
runs the real trainer end to end on a float16 toy at both patch settings.

### Also fixed before this dispatch

`truth_pentad.npz` existed only in a previous session's sandbox, so #365 built
its tensor with **no transport labels at all** and the probe died on
`KeyError: 'rapid'`. It is now published on the Hub and the workflow's fetch is
**fatal** — a tensor with no labels is worse than a failed build. The labels
themselves are the cadence dividend, measured: **RAPID 1,459 pentad labels
against 240 monthly (6.1×)** and **Florida Current 2,553 against 433 (5.9×)**,
means physically right at 17.0 Sv and 31.8 Sv.

### Attempt 1 (2026-08-17 ~18:20Z): both arms dead, one of them GREEN

- **#366 (f4-40M, `gpu-box-47094145`) — VOID, and it reported SUCCESS.** Its
  first training step carried ~537 s of one-time cost (first CUDA kernels,
  first touch of the 33 GB tensor). The `--max-minutes` calibration fired at
  `elapsed > 60 s` — i.e. at step 1 — read 537.54 s/step against a true steady
  rate of **0.19 s/step**, and re-fit the cosine schedule from 200,000 steps to
  **66**. The run trained 66 steps, annealed the LR to zero, saved, passed
  every probe and went green with 691 of its 700 budgeted minutes unspent.
  **Do not quote any number from #366's probe archive** — its codec is
  effectively untrained. Fix: `fit_schedule` (rate excludes step 1, minimum
  3-step sample, periodically re-checked and allowed to grow back), pinned by
  `tests/test_max_minutes_refit.py`, which replays #366's exact numbers.
- **#367 (f4-200M, `gpu-box-46996216`) — CANCELLED at 4 min**, deliberately:
  the box still held #365's tensor, built before the labels were published —
  same recipe string, no labels — so the recipe guard skipped the rebuild and
  the run would have died 20 h later in `probe_kfold` on `KeyError: 'rapid'`.
  The guard now verifies the labels themselves (`missing_truth_keys`).
- **#368 (f4-200M re-dispatch, same box) — exit 137 at 26 min.** Host OOM:
  `gpu-box-46996216` has **63 GB** RAM against X's 33 GB decompressed plus the
  anomaly transform's float64 temporaries. #366's preamble survived only
  because its box has 126 GB. Interim rule until the memmap reader lands:
  **family-4 jobs go on 126 GB boxes only.**

One incidental positive: #368's build step is direct evidence the truth guard
works — it found the label-less cached tensor and rebuilt with labels in
12 min.

### Attempt 2 (2026-08-18 ~06:15Z): re-dispatch on 126 GB boxes, plus the control

- **f4-40M** → **#386** on `gpu-box-47094143` (126 GB)
- **f4-200M** → **#387** on `gpu-box-47094145` (126 GB, warm labeled tensor)
- **FROZEN CONTROL** → **#388** on `gpu-box-47529389` (126 GB), then
  **#390** (attempt 2, same box, warm tensor) after #388's probes all died —
  see below:
  `resume: !f3_anchor41M` at `--steps 60000` = the checkpoint's own recorded
  step, so the loop never runs — a cross-tensor EVAL, newly allowed by the
  resume guard exactly when nothing will train
  (`tests/test_frozen_control_resume.py` pins that the "frozen" control's
  saved weights are bit-identical to the loaded anchor). The released
  `f3_anchor41M__pixelmae.pt` was **opened and read** (ml/CLAUDE.md §0.1):
  step 60000, tag run-80, d_z **64**, **patch 3**, 576/10/8/768 = 40,692,849
  params, `data=family3_na025.npz`, same holdouts as family 4's defaults.
  Its `probe_kfold` on the family-4 tensor is the baseline both trained arms
  are reported against, and the number that can falsify E-038's premise.

### #388 (frozen control, attempt 1): the eval mechanism worked; the probes died where #365 did

**The cross-tensor eval path ran exactly as designed** — `CROSS-TENSOR EVAL:
codec trained on family3_na025.npz, evaluated on family4_na025_pentad.npz. No
training will occur (checkpoint step 60000 >= --steps 60000)`, then
`checkpoint is already at/past --steps; nothing to do`. Zero steps trained.
**And then all three probes were OOM-killed**, each ~30 min in — immediately
after the anomaly transform, on the same full-size `isfinite` bool that
killed #365's trainer (16.6 GB at pentad; 83 GB at daily, i.e. impossible for
#389, the daily arm). The run went green; the only trace was the workflow's own warning
`no probe_kfold.json — this bundle has no CODEC control`. Fixed by giving
probe_kfold / probe_sequence / dip_check the LazyPixels treatment
(embeddings pinned bit-identical, `tests/test_probe_lazy_pixels.py`);
re-dispatched as **#390** (the frozen control, attempt 2).

### E-038c (2026-08-18 ~07:00Z): the daily arm's first run

**#389 (f5-40M, `gpu-box-46996216`, the 700 GB box)** — the first run of the
whole daily pipeline: `family5_na025_daily` built on the box in 52 min
(sidecar layout, 165.6 GB memmappable `_X.npy`), memmapped training with the
scratch-copy anomaly transform, centred 5-day rolling wind σ. Budget-matched
to the pentad arms at 200k × 512; 37,975,889 params against a ~441 M daily
Chinchilla anchor — deliberately far under; the capacity rung (E-038d, 200M)
waits for a second 600 GB+ box and for this run to validate the pipeline.
Labels: `truth_daily.npz` — **RAPID 7,290 daily labels = 30.4× monthly, FC
13,931 = 29.2×** (the ~30× prediction, confirmed).

### THE FROZEN-CONTROL BASELINE (#390, completed 2026-08-18 10:10Z)

The number E-038 is measured against, from `probe_kfold` on run-80's monthly
anchor evaluated frozen on the pentad tensor — log verbatim:

> `rapid  k-fold r +0.660 [+0.593, +0.722]  (n=1459) · RMSE 2.97 Sv (sigma
> 3.95) · 18mo-lowpass r +0.582 · wind-only +0.670 [+0.601, +0.733] (RMSE
> 2.93)`

Dip check: 2009–10 event 45% captured, out-of-fold r +0.660, sign agreement
71%.

**Reading, in three parts:**

1. **At pentad, the frozen monthly codec does NOT clear the wind bar.** The
   codec reads 0.660 against wind-only 0.670 — indistinguishable (shared
   folds; a paired test would be needed for the sign, per §3), and clearly
   not ABOVE it. At monthly the same codec cleared the same bar by +0.063
   (0.631 vs 0.568). Whatever the anchor's embeddings add beyond wind at
   monthly, they add nothing at pentad — the direct, quantitative form of
   the out-of-domain premise. **The bar for E-038a/b is therefore 0.670**,
   the pentad wind baseline, not 0.660.
2. **The wind floor itself rose from 0.568 (monthly) to 0.670 (pentad).**
   Physically sensible: at 5-day resolution RAPID transport variability is
   increasingly Ekman/wind-driven, so raw wind stress explains more of it.
   Finer cadence raises the floor the codec must beat — more labels, but a
   harder null.
3. The trainer's crude in-train `rapid_probe` returned NaN on the resumed
   codec (both #388 and #390, the two frozen-control attempts) while the k-fold ran cleanly — an instrument
   nit to chase, not a result.

**Two defects #390 exposed:** `probe_sequence` was STILL OOM-killed (its
`d["X"].copy()` holds the decompressed tensor twice — residual, recorded, off
the critical path); and the k-fold silently carried **no FC entry**:
`target_series` decoded family-4 truth arrays as (YYYYMM, value) monthly
pairs, matched 0 of 2,553 row-indexed FC labels, and returned None with
nothing in the log. Fixed (row-decode when the tensor carries `bin_index`);
the FC baseline needs one cheap re-probe, which can ride any later eval
dispatch.

**In-flight at 11:15Z:** #386 (E-038a, 40M) step 42,000, light probe 0.617 and
climbing, ~0.23 s/step, no refit. #387 (E-038b, 200M) step 15,000 — early z-space
EXPANSION (step-7500 full probe: z_mse_persistence overflowed to Infinity,
linear probe dipped 0.54 → 0.32 → 0.39 recovering; losses and temporal_r
healthy) — the §4.10 two-sided-guard story, watched, not yet acted on.
#389 (daily) ~4 h into its preamble with no metrics yet — expected: the
per-channel anomaly transform over a 166 GB memmap pages in the whole file
per channel, ~13 TB of I/O; silence until ~12:30Z is consistent with health.

**That last sentence was wrong, and the correction is the reason for the two
entries below.** #389 was CANCELLED at ~7 h, still in the same function: the
per-channel loop charges **249.8 full-extent traversals**, so 165.6 GB on a
64 GB box is ~41 TB of physical read, not 13. `anomaly_transform` is now
time-chunked at 6.0 traversals (commit `fba358c`; 40.4 TB → 994 GB, measured
end to end 1.95 h → 352.8 s on a 7.46 GiB float16 memmap, bit-identical output
over 3,182,755 entries). "Consistent with health" was a model of the code
standing in for a measurement — ml/CLAUDE.md §4.12.

### AUDIT (2026-08-18): two more copies of the anomaly transform, and what they touched

`ml/temporal.py` and `ml/probe_sequence.py` each carried a hand-inlined THIRD
and FOURTH copy of the transform, both frozen at the **pre-2026-08-17**
arithmetic — `v.std()` with no `dtype=np.float64`. On a float16 tensor that
sums ~204M squared residuals past 65504, returns `inf`, and
`(X - mu) / (inf + 1e-6)` is **exactly 0.0**: every dynamic channel becomes
zeros while every loss, `gpu_util` and probe still reads healthy. Families 4
(pentad) and 5 (daily) are float16. Measured on a shared fixture: the inlined
copy returns sd 0.000000 with 100.0% of entries exactly zero at float16, and
sd 1.012848 with 0.0% at float32. Both copies also carried the 249-traversal
shape, so stage 2 on the daily tensor would have reproduced the daily arm #389's hang.

**Which results on record came through those two copies on a float16 tensor:
NONE.** The audit, over every `ml-train.yml` run #1–#396 (workflow logs plus
the `ml-metrics` archive's `probes-<n>.json` / `provenance.json`):

| run | tensor | dtype | `temporal_steps` | what the two scripts produced |
|---|---|---|---|---|
| #365 f4-40M | family4_na025_pentad | f16 | 0 | `probe_sequence` died at `torch.load` (no checkpoint — the run OOMed first); no `temporal.json` |
| #366 f4-40M (VOID) | family4_na025_pentad | f16 | 0 | `probe_sequence` **OOM-killed** in the transform; no `temporal.json` |
| #367 f4-200M | family4_na025_pentad | f16 | 0 | cancelled at 4 min; `probe_sequence` died at `torch.load` |
| #368 f4-200M | family4_na025_pentad | f16 | 0 | exit 137; `probe_sequence` died at `torch.load` |
| #388 frozen control | family4_na025_pentad | f16 | 0 | `probe_sequence` **OOM-killed**; no `temporal.json` |
| #389 f5-40M (daily) | family5_na025_daily | f16 | 0 | cancelled; `probe_sequence` died at `torch.load` |
| #390 frozen control | family4_na025_pentad | f16 | 0 | `probe_sequence` **OOM-killed**. Its `probe_kfold` **0.660** and `dip_check` came through `trainprobe.anomaly_transform`, which had the float64 fix at that sha (`70ffe2d` contains `2752b8b`) — **the E-038 baseline stands** |
| #391 read-out ladder | family4_na025_pentad | f16 | — | failed at the tensor step (exit 1); no probes |

Not one of them archived a `probe_sequence.json` or a `temporal.json`. Every
family-4/5 dispatch to date carries **`temporal_steps: 0`**, so
`ml/temporal.py` has never executed against a float16 tensor at all.

Everything that DID run stage 2 — the wave-8 heads and the E-035/E-036/E-037
arms — ran on **`family3_na025`, which is float32** and never reaches the
limit: #350, #351, #357, #358, #359, #360, #363, #364, #395, and the eval
runs #352–#356, #369, #380–#382. Provenance read from
`probes-<n>.json:provenance.json` for each; #357/#363/#364 from their live
`metrics.jsonl` and job logs (#357's log has expired). **No published number
in this file is affected.**

**NOT DETERMINABLE at the time of the audit**, because GitHub does not serve
logs for a run in progress and neither had archived provenance yet:

- **#392** (E-038 read-out ladder on the frozen anchor, `family4_na025_pentad`,
  phase "probes and stage 2" at 13:32Z) — **RESOLVED at 15:40Z**, when
  `probes-392.json` landed on `ml-metrics`: `temporal_steps: 0`,
  `head_probe: true`, `resume: !f3_anchor41M`, and the bundle carries
  `probe_kfold.json` + `dip_check.json` and neither a `temporal.json` nor a
  `probe_sequence.json`. **Unaffected**, on the same reasoning as #390.
- **#386** (E-038a f4-40M, `family4_na025_pentad`) — STILL OPEN at 15:40Z:
  124 metric lines, no stage-2 line, no archived provenance, so its
  `temporal_steps` is unread. It is stage-1 training and has not reached its
  probe ladder. It will run the code it CHECKED OUT AT JOB START — sha
  `c7ba151`, 06:02Z — which still carries both broken copies; this commit
  cannot reach it. **Read `probes-386.json` when it lands and confirm
  `temporal_steps`. If it is non-zero, that run's stage-2 number came through
  the broken copy on a float16 tensor and must be discarded, and the run will
  also have spent hours in the 249-traversal transform.**
- **#393 / #394 / #396** (the E-036, E-037 and E-035 evals) are `sroll:`/family-3 and unaffected.

Both copies are now calls to `trainprobe.anomaly_transform`, and the class of
defect is pinned rather than fixed one file at a time:
`tests/test_one_anomaly_transform.py` fails if ANY file under `ml/`
re-implements the transform, and `tests/test_stage2_float16_anomaly.py` runs
both scripts' real transform path on a float16 fixture in the overflow regime
(train pool 746,712; the old arithmetic reads `inf`) and asserts the dynamic
channels are not zero — verified to FAIL on the pre-fix code at 100.0%
exactly-zero for both scripts.

One more copy exists and is deliberately left: `ml/recon_eval.py` carries a
STREAMING replica (its docstring says so; the in-RAM recipe needs >11 GB) with
its own `verify_streaming` cross-check. It cannot hit this bug — every
reduction in it is float64 or preceded by `.astype(np.float32)` — and it has
never run in any workflow run. It is named in the test's exemption list with
that reason. **Separate task:** its climatology counter is
`n = np.zeros(..., np.uint8)`, sized for "≤43 train months per moy"; that
wraps silently at family 4's ~262 and family 5's ~1309 timesteps per
month-of-year, so it must not be pointed at a pentad/daily tensor as it
stands.

**Also separate:** `ml/probe_head.py` and `ml/dip_check.py` still open the
tensor with plain `np.load`, so they cannot read a family-5 sidecar tensor at
all. That failure is a `KeyError`, i.e. loud, and both already call the
canonical transform — so it produces no wrong number, only a missing one. Left
untouched here deliberately: neither can be exercised end to end in the
sandbox, and an unverified change to two more probe scripts on the eve of an
eval wave is the wrong trade. **RESOLVED the same evening** by #392's OOM (below):
both now open the tensor through `tensor_io.load_tensor` + a scratch
`writable_copy`, taken after the channel guard and removed at exit, so the
canonical map never takes the transform's writes.

### Run ledger, 2026-08-18 evening — two runs that produced no number

- **#389 (f5-40M, daily) — CANCELLED at ~7 h**, having never left
  `anomaly_transform`. The per-channel loop charged **249.8 full-extent
  traversals** of a 165.6 GB memmap on a 64 GB box: ~41 TB of physical read,
  not the 13 TB the in-flight note guessed. The transform is now time-chunked
  at **6.0 traversals** (40.4 TB → 994 GB; measured end to end 1.95 h →
  352.8 s on a 7.46 GiB float16 memmap, output bit-identical over 3,182,755
  entries). Nothing was trained; the daily arm is unmeasured.
- **#392 (read-out ladder on the frozen anchor, pentad) — its `probe_kfold`
  landed, both `probe_head.py` invocations were OOM-KILLED**, and the job
  still reported SUCCESS because both steps are best-effort. So the one
  number Chris says he trusts at this cadence — the unpooled head and its
  matched raw-3×3 control — does not exist for #392. The peak was **132.5 GB**:
  the 33.1 GB tensor, a 16.6 GB `isfinite`, and an **82.8 GB transient inside
  `np.nan_to_num(..., copy=False)`**, which allocates five full-size bools at
  once (`isnan`, `isposinf`, `isneginf`, and the `isinf`/`signbit` underneath
  the last two) — `copy=False` promises no copy of the VALUES, not no
  allocation. `probe_head` and `dip_check` got the LazyPixels treatment they
  were missed by in the #388 (frozen control) round (measured on a 0.523 GiB fixture: VmHWM
  2.132 → 0.642 GiB, 3.3×, with the eager path pinned as a tripwire).
  **Re-dispatched as #397** (read-out ladder, attempt 3) to produce the head number.

### #397 (E-038 read-out ladder, attempt 3) — the head probe died a THIRD time, and the cause was neither memory nor code: THE BOX HAS NO C COMPILER

**The memory fix held.** `f2ee8b8`'s LazyPixels treatment worked exactly as
measured: #397's embedding completed in **~2.5 minutes at low RAM**, where
#392 (the ladder's previous attempt) had been OOM-killed in `np.nan_to_num`. Nothing about the 132.5 GB
transient recurred.

**Then the FIRST `loss.backward()` in `fold_fit` (`ml/probe_head.py:102`)
raised**, verbatim from the job log of run #397 (id 32174255568):

```
File ".../torch/_native/ops/bmm_outer_product/triton_impl.py", line 28,
  in _bmm_outer_product_impl
File ".../triton/runtime/build.py", line 32, in _build
RuntimeError: Failed to find C compiler. Please specify via CC environment
  variable or set triton.knobs.build.impl.
```

Both invocations died identically — the head at **20:14:54** and its matched
raw-3×3 control at **20:26:23**. So for the third time in one day, and for
the third distinct reason, the run that exists to produce the unpooled head
number produced provenance and a `probe_kfold` and nothing else. (#397's
`probe_kfold` did land, and reproduces #392 exactly: rapid
**0.660** [0.593, 0.722] against the wind-only ridge bar of **0.670**
[0.601, 0.733].)

**Mechanism.** torch 2.13's `_native` eager router dispatches this backward
through a **triton** kernel, and triton **JIT-compiles C at first use**. The
Vast box image ships no `cc`. It is a dispatch-table property of the op, not
of our code — which is why it bites `probe_head` and has never bitten
anything else here: an eval-only run does no codec backward at all, and
`probe_sequence`'s temporal-transformer backward does not route through this
op. Whether the codec-TRAINING runs (e.g. #386, the E-038a codec) miss the op or merely happen
to sit on boxes that carry a compiler is **not established and does not
matter for the fix**.

**Fix (this commit).** The `Install deps` step of `.github/workflows/ml-train.yml`
now ensures a C compiler exists before anything that can backprop:
install-if-missing (`command -v cc || command -v gcc` → `apt-get install -y -qq
gcc`; the boxes run as root, same as `gpu_box.mjs`'s onstart), then resolve the
binary and export **`CC` through `$GITHUB_ENV`**, so the **Train** step and the
**probe ladder** both inherit it — a future codec architecture that happens to
route through triton must not die at step 1 of a fourteen-hour run. A warm box
skips the apt entirely in well under a second; `ubuntu-latest` never enters the
branch. It is deliberately **not fatal** (§5.17 — most runs never touch a
triton path, and killing them for a transient apt mirror costs more than it
protects), but it **asserts the effect** (§0.2): it re-checks for the binary
after the install and prints its version, so "installing gcc" can never be a
log line about nothing. Both branches were exercised locally under `bash -e`
before the push — compiler-present (exits 0, writes `CC=/usr/bin/cc`) and
compiler-absent with a stub `apt-get` (warns, exits 0, writes nothing).

### #386 (E-038a, the f4-40M pentad codec) — the run COMPLETED and all three of its probes OOM-died on code that predated the day's fixes

**#386 (E-038a, f4-40M, the first codec trained from scratch on the r1 pentad
tensor) finished successfully at 22:04Z.** It was dispatched at **06:02Z**
pinned to sha `c7ba151` — before **every one** of 2026-08-18's probe fixes —
so `probe_sequence` (21:10), `probe_kfold` (21:38) and `dip_check` (22:03)
were **all OOM-killed**, and `probes-386.json` carries **provenance only**.
This is the outcome the audit above predicted in writing while the run was
still open ("it will run the code it CHECKED OUT AT JOB START — sha
`c7ba151`, 06:02Z — which still carries both broken copies; this commit
cannot reach it").

**The trained checkpoint is safe.** `Upload checkpoint + eval` was green
(artifact `pixelmae-386`, 323 MB, ID 9342241162, 21:00Z). Its training ran
under `max_minutes: 850`, and `fit_schedule` re-fitted the cosine repeatedly —
the last re-fit line reads *"re-fitting … from 158766 to 166752 steps"* — and
the run trained that annealed schedule **to completion**:
`run-386.jsonl` on `ml-metrics` ends at **step 166,752** (`loss_rec` 0.22035,
`loss_nei` 0.19750, `wall_s` 48,849.1), and the log's final progress line is
`step 166750/166752`, followed by `saved ml/runs/actions/pixelmae.pt`.
`train.py:897` assigns the re-fit total back into `a.steps`, and the final
save is `save_ckpt(a.steps)` (line 1030), so the checkpoint's **recorded step
is 166,752**, not the dispatched 200,000.

**Where the checkpoint is.** There is **no** `run-386` asset on the
`model-checkpoints-v1` release — codec checkpoints reach that release only via
the rescue path, and `rescued-orphan-latest-386.pt` (06:05:55Z) is the file
#386 *rescued from the previous job on its box*, not its own. #386's own
checkpoint lives in two places: the Actions artifact `pixelmae-386`, and the
box-persistent mirror `/opt/earth-cache/ckpt/run-386.pt` written by
`save_ckpt` under `CKPT_TAG=run-386`. That mirror is intact on
**gpu-box-47094143**: the mirror never reported a skip, `disk_hygiene.sh`
exited at its first check on that job (`disk hygiene: 44 GB free, want 16 GB`)
so it pruned no `run-*.pt`, its prune rule keeps the two highest run numbers
in any case, and **no job has run on that box since**. Hence the eval below
resumes `!run-386`.

**In-training numbers #386 did produce** (light probe at its final step, NOT
`probe_kfold` and not comparable to the headline bar): `linear_r_deseas`
**0.624**, `linear_r_raw` 0.646, `temporal_r_deseas` 0.604,
`chan_vs_persistence` +31.8%, `z_vs_persistence` +43.3%.

### Two EVAL-ONLY dispatches, 2026-08-18 ~22:20Z — #406 and #407, both QUEUED AGAINST A STOPPED BOX

Both carry the `cc` guard above and both are dispatched on `ref: main` at
`c215ba7`, the commit that introduced it — verified on the runs themselves,
`head_sha == c215ba7d0244a14b7f89b340fbca1f6591137e12` for each.

**They are queued and will not start.** Both were dispatched on the premise
that their target boxes were RUNNING and idle-burning, so that the evals were
free in the marginal sense. **That premise was false at dispatch time**, and
it was measured after: `/api/v1/instances/` reports vast **47720664**
(`gpu-box-47529389`) and **47720660** (`gpu-box-47094143`) both
`actual_status: exited`, `cur_state: stopped`, and GitHub independently
reports both runners **offline** — two independent sources agreeing.
`scripts/fleet_health.mjs` reads **"0 runner(s) online+idle"** across a fleet
of six boxes, all of them busy with other work. Neither box was stopped by
this task; the hourly fleet-health check had already stopped them after #397
(the ladder's third attempt, 20:26Z) and #386 (E-038a, 22:04Z) finished. **No box was started or stopped here**,
so the two runs sit queued until one is, at which point the pinned `runner`
input routes each to the box that holds its warm tensor and checkpoint.
Queued jobs cost nothing. Costs so far: **zero GPU-seconds**.

**Eval A is #406** ([run 32192203050](https://github.com/blauewelt/earth/actions/runs/32192203050)),
**Eval B is #407** ([run 32192246793](https://github.com/blauewelt/earth/actions/runs/32192246793)).

**Eval A — the anchor head number, ATTEMPT 3.** On `gpu-box-47529389` (vast
47720664; warm: r1 pentad tensor + `f3_anchor41M`). #397's verbatim inputs,
all 25, `doc` alone changed: `steps` 60000 (= the checkpoint's own step, so
nothing trains), `resume` `!f3_anchor41M`, `head_probe` true, `tensor`
`family4_na025_pentad`, `patch` 3, codec 576/10/8/768, `d_z` 64,
`temporal_steps` 0, `light_probe_every` 0, `eval_every` 7500, `window` global,
`anomaly` true, `max_minutes` 0, `job_timeout` 350, `lr_floor` 0,
`lr_decay_steps` 0, `batch` 512, `sst_channel` false, `runner`
gpu-box-47529389.

*What it must produce:* the **unpooled head number** — cross-attention over
the ~67 raw 26.5°N section pixel embeddings — and its **matched raw-3×3
end-to-end control**. *Falsifier, unchanged since #392, the ladder's second attempt:* if the head does not
clear the wind-only ridge bar of **0.670**, E-038's pentad headline is
**representation-limited** (the pentad codec has not learned transport); if it
does clear it, the pooled decline is a **read-out artefact** of
`probe_kfold`'s `Z.mean(1)`, which annihilates the east-minus-west contrast
across 26.5°N that geostrophic transport *is*. Three attempts, three distinct
deaths: **#392** the 82.8 GB `nan_to_num` transient, **#397** the missing C
compiler, and before them #391 the tensor step.

**Eval B — #386's own codec, full probe ladder + head.** On
`gpu-box-47094143` (vast 47720660; warm: r1 pentad tensor + `run-386.pt`).
Eval-only by the same mechanism: `steps` **166752** = the checkpoint's recorded
step, so `train.py`'s `while s < a.steps` never turns over. Codec geometry
**matches #386 exactly** — 512/12/4/256, `d_z` 32, `patch` 1 — because
`--resume` derives architecture from the checkpoint's own `args` and a
contradicting dispatch is refused (this is the #395 failure — E-035 seed 0, attempt 1 — sixty
`size mismatch` lines in 90 s). Inputs: `resume` `!run-386`, `head_probe`
true, `anomaly` true, `temporal_steps` 0, `light_probe_every` 0, `eval_every`
7500, `tensor` `family4_na025_pentad`, `window` global, `sst_channel` false,
`max_minutes` 0, `job_timeout` 350, `lr_floor` 0, `lr_decay_steps` 0, `batch`
512, `runner` gpu-box-47094143.

*What it must produce:* #386's **`probe_kfold`** — the number the whole E-038a
arm was trained to generate and which has never existed — against the
wind-only ridge bar of **0.670** and the frozen-anchor bar of **0.660**, AND
its **unpooled head number** on the fixed code. *Falsified* if the
from-scratch pentad codec does not beat the frozen monthly anchor: that is
E-038a's original out-of-domain hypothesis, and 21 hours of training have so
far bought no measurement of it at all.

**Eval B update, 2026-08-19 01:22Z (hourly fleet-health check): #407 CANCELLED, re-dispatched as #409 on `gpu-box-47529389`.** #406 finished green at ~01:0xZ and left its box online+idle (the check's IDLE BURN flag); #407's pinned box `gpu-box-47094143` (vast 47720660) was still `exited` with `start` returning `resources_unavailable` — three hours after the first attempt, so the handoff's fallback applied. #409 carries #407's inputs verbatim except `runner: gpu-box-47529389` and the doc line; `resume` stays `!run-386`, satisfied on the new box by the "Seed resume checkpoint from the release" step pulling `run-386__pixelmae.pt` (455.9 MB, verified on `model-checkpoints-v1`) — the r1 pentad tensor is already warm there from #390/#392/#397/#406. Dispatched on current `main` (`c4c900c`, carries the `cc` guard and the bounded-rescue fix). Picked up in under 90 s. A `stop` was issued to vast 47720660 to cancel its queued start intent, so it does not come up idle later; nothing unique remains on that disk (`run-386.pt` is on the release).

**CORRECTION, 2026-08-19 ~01:45Z — "they are queued and will not start" was
wrong within three minutes of being committed.** The section above is left
standing rather than rewritten, because the fleet reading it was written from
was honest and correctly sourced; but the world moved under it. The main session
started Vast **47720664** at **~22:22Z** — two minutes BEFORE the commit
(`409b2c0`, 22:24:37Z) — and **#406 was picked up at 22:27:25Z**, three minutes
after it. #406 then ran to **success at 00:24:48Z**. So **"zero GPU-seconds" was
true for about seven minutes** and is false as a record of what this pair cost:
#406 spent ~2 h of a $0.29/h box and produced the first `probe_head.json` in
four attempts — the best money in the wave. Only the #407 half of the claim
held: its box's host stayed full, it never started, and it was cancelled at
01:22Z and re-dispatched as #409. The lesson is not that the measurement was
bad but that **a cost claim written in the present tense expires** — state it
as "as of HH:MMZ", or state it after the run ends.

### RESOLVED, 2026-08-19 00:24:48Z — #406 (the read-out ladder, attempt 3) landed, and E-038's pentad headline was a READ-OUT ARTEFACT

**Run #406** (`head_sha` `c215ba7`, archived as `probes-406.json`): the frozen
`f3_anchor41M` monthly anchor, scored on the r1 pentad tensor, **n = 1459**, the
same year-blocked k-fold protocol on every row.

| read-out | rapid r (k-fold, deseasonalised) | 95% CI | RMSE (Sv) |
|---|---|---|---|
| pooled ridge on `Z.mean(1)` | **0.660** | [0.593, 0.722] | 2.97 |
| **attention head over the ~67 UNPOOLED section tokens** | **0.691** | [0.631, 0.746] | 2.87 |
| matched raw-3×3 end-to-end control (same head architecture, raw pixels, no codec) | **0.683** | [0.620, 0.742] | 2.89 |
| wind-only ridge bar | 0.670 | [0.601, 0.733] | — |

`fc`, the same protocol: pooled **0.395** [0.300, 0.487] against a wind-only
**0.199** [0.129, 0.271].

**(a) The pooled read-out was destroying real signal.** Unpooling alone is worth
**+0.031**, and the head is the ONLY read-out here that clears the wind-only bar
— **0.691 against 0.670**, where the pooled ridge sat *below* it at 0.660.
Chris's standing directive — distrust pooled read-outs, geostrophic transport is
the east−west contrast ACROSS the section and a mean annihilates it — has been
an argument since E-034; it is now a measurement. Every "does not clear wind"
headline this programme has written off a `probe_kfold` number over a section
tensor is suspect for exactly this reason.

**(b) But the anchor's EMBEDDING adds ≈ nothing over raw pixels through the same
head.** 0.691 against the raw control's 0.683: **Δ = +0.008**, with CIs that
nearly coincide ([0.631, 0.746] against [0.620, 0.742]) — far inside noise at
n = 1459. Read plainly: at pentad cadence, what the head reads out is already
present in the raw fields, and the frozen monthly representation contributes no
measurable additional transport information. The gain that looked like the
codec's turns out to be the read-out's.

**(c) E-038's pentad question therefore decomposes, and only half of it is still
open.** The earlier headline — "the pentad arm does not clear wind" — was a
READ-OUT artefact and is withdrawn as a statement about representation. What
remains open is the sharper question the raw control makes askable at all: **does
any TRAINED pentad codec's embedding beat its OWN raw control through the same
head?** That is precisely what **#409** measures (#386's from-scratch pentad
codec — **LANDED, and the answer is no: 0.680 vs 0.683, see the VERDICT section
immediately below**). #409's head ≈ its raw control too, so **stage-1
pretraining is not paying at pentad in the read-out we now trust** —
and **E-042** (SST) and **E-038c** (daily) must be judged on this same protocol:
head against matched raw control, not head against wind.

**(d) Provenance — the `cc` guard did real work.** The box had **no C compiler at
all**; the guard installed gcc in **~9.5 s** (the entire "Install deps" step took
15 s). `probe_head.json` exists for the **first time in four attempts**: **#388**
OOM, **#392** the 82.8 GB `nan_to_num` OOM, **#397** triton-no-cc, **#406**
success. Three of those four deaths were instrumentation and none were science,
and each was diagnosed only after it had already spent the GPU — the argument
for §0 rule 3 (guard at dispatch, where the inputs are all it has cost you).

### VERDICT, 2026-08-19 ~03:30Z — #409 (E-038a's own codec through the full probe ladder): the trained pentad codec does not beat its own raw pixels

The other half of the question the section above left open. **Run #409**
(`head_sha` `c4c900c`, archived as `probes-409.json`) reads #386's OWN codec —
**37,975,889 params**, trained from scratch on the r1 pentad tensor at **batch
512** for **166,752 steps** over ~14 h, on the **191,520,806-pixel-pentad**
train pool (E-038a's four scale numbers, above) — through the identical #406 (read-out ladder)
protocol: eval-only, **n = 1459**, year-blocked k-fold on every row.

| read-out | rapid r (k-fold, deseasonalised) | 95% CI | RMSE (Sv) |
|---|---|---|---|
| pooled ridge on `Z.mean(1)` | **0.652** | [0.582, 0.719] | 3.00 |
| **attention head over the UNPOOLED section tokens** | **0.680** | [0.617, 0.740] | 2.91 |
| **matched raw-3×3 end-to-end control** (same head architecture, raw pixels, no codec) | **0.683** | [0.620, 0.742] | 2.89 |
| wind-only ridge bar | 0.670 | [0.601, 0.733] | 2.93 |

`fc`, same protocol: pooled **0.390** [0.316, 0.465] against a wind-only
**0.199** [0.129, 0.271]. `dip_check` captures **40.6%** of the 2009/10 dip
(−2.82 Sv predicted of −6.95 observed), sign agreement 72.4%.

**(a) The trained codec's head does not beat its own raw control.** **0.680
against 0.683 — Δ = −0.003**, the wrong sign, and far inside a CI ~0.12 wide.
Fourteen hours of from-scratch pentad pretraining produced an embedding that is
**indistinguishable from raw pixels** through the read-out this programme now
trusts. This is the cleanest form the comparison can take: same tensor, same
head architecture, same folds, same n, differing only in whether a codec sits
between the pixels and the query.

**(b) It does not beat the frozen monthly anchor either.** Head **0.680** vs the
anchor's **0.691**; pooled **0.652** vs **0.660**; dip capture **40.6%** vs
**45.2%**. E-038's central hypothesis — *pentad fields are out of domain for the
frozen monthly codec, so a codec trained ON pentad beats it* — is **FALSIFIED in
the head read-out**. The out-of-domain anchor is, if anything, slightly better,
and it is better on all three rungs at once, which is harder to dismiss as one
noisy number than any single row would be.

**(c) The consolidated picture at pentad cadence.** All three read-outs that see
UNPOOLED section pixels — frozen-anchor head 0.691, trained-codec head 0.680,
raw-pixel control 0.683 — land in a **0.680–0.691** band with CIs that overlap
almost completely, barely clearing the wind-only bar of **0.670**. **No
embedding, frozen or trained, adds measurable transport information beyond the
raw fields.** What the head reads is in the PIXELS, not in the representation;
the only thing that moved the number in this whole ladder was unpooling, i.e.
read-out design.

**(d) Consequences for the programme, stated explicitly so they bind future
dispatches.**

1. **Any future stage-2 comparison must carry a RAW-INPUT control arm**, not
   just embedding arms. On the numbers above the honest prior is that raw
   would MATCH — an embedding-vs-embedding matrix cannot distinguish "this
   representation is good" from "both representations are the pixels".
2. **E-042 (SST) remains worth running**, because a new channel changes what
   the pixels contain rather than how they are encoded. But its decisive
   read-out is the **head + raw-3×3 pair on the r2 tensor** — *does SST move
   the RAW ceiling?* — not the codec probe alone. An r2 codec probe that beats
   r1's tells us nothing we could act on if the raw control moves with it.
3. **The deferred embedding-vs-embedding stage-2 matrix** (arm A `!run-386`
   against arm B `!f3_anchor41M`, held at the top of this file on credit) is
   now **DOWN-PRIORITISED ON EVIDENCE, not merely on credit.** Its two arms are
   the two representations just measured as equivalent-to-raw and to each
   other. Re-scope it around a raw arm before it is re-dispatched.

**(e) ADDENDUM, 2026-08-19 ~11:00Z — the measurement stands; consequence (d.3)
was drawn too narrowly and is REVERSED.** Chris, on the programme's object:
*"What we're building is a predictor for everything, so it doesn't matter
whether the embedding will contain more or less information than the raw pixels.
The embedding makes large chunks of data 'attendable' by a transformer. And we
can predict everything from predicting embeddings (not just AMOC). That's the
overall plan."*

Nothing in (a)–(c) changes. The numbers are what they are: head **0.680** vs raw
**0.683** vs frozen anchor **0.691**, three read-outs inside a 0.011 band, and
E-038's domain-shift hypothesis is still falsified in the head read-out. What
changes is what those numbers are evidence ABOUT.

1. **(d.1) and (d.2) STAND, as READ-OUT controls.** A raw-input arm remains
   mandatory wherever a current-state probe is being quoted, and E-042's
   decisive read-out is still the head + raw-3×3 pair on the r2 tensor. That is
   exactly the job E-038a's raw control did correctly — it exposed the pooling
   artefact and it keeps "the codec knows this" separable from "any read-out
   with spatial structure knows this". Keep it.
2. **(d.3) is REVERSED. The embedding-vs-embedding stage-2 matrix is
   RE-PRIORITISED** (arm A `!run-386` against arm B `!f3_anchor41M`, temporal
   384×6 @ 24k steps, `head_probe` on both). *Which embedding rolls forward
   better* is the programme's actual question, and it is the one question the
   #409 ladder does not touch. A current-state probe asks what today's
   embedding says about today's transport; the forecaster never has to answer
   that. Probe parity between two representations is therefore **not** evidence
   that they are equivalent as forecast substrates, and (d.3) treated it as if
   it were.
3. **The primary embedding metric is corridor AUC from ROLLED-FORWARD
   embeddings**, plus rollout skill and band correlations — the instruments
   E-022, E-035, E-036 and E-037 already score. Judge a representation by what
   stage 2 can predict from it, not by a probe on the un-rolled state.
4. **A raw-pixel head is a control, not a rival architecture.** There is no
   raw-pixel forecaster to re-scope the matrix around: 165.6 GB of daily pixels
   is not attendable at any batch size on any box in the fleet, which is the
   tractability problem the codec exists to solve. "Re-scope it around a raw
   arm before it is re-dispatched" in (d.3) asked for an arm that cannot be
   built at this cadence and resolution; the raw arm belongs in the read-out
   ladder, where it already is.

Standing form of this in `ml/CLAUDE.md` § "What this programme is building" and
`docs/ML_BASICS.md` §1b.

### E-038c ATTEMPT 2 (2026-08-18 20:35Z): #400, the daily arm re-dispatched

**Hypothesis, unchanged from #389, this arm's first run.** A 38 M codec trained from scratch on the
DAILY family-5 tensor beats the frozen monthly `f3_anchor41M` applied to that
same tensor — E-038a/b's domain-shift claim one cadence finer, where E-034 §4's
one-live-RG-bin-per-month policy pushes the `missing`-token share to **96.7%**
of daily bins against ~83.6% at pentad.

**Control.** `f3_anchor41M` (run #62, 40.7 M) scored by `probe_kfold.py` on the
family-5 tensor, plus the pentad row from E-038a. **Falsified** if the fresh
daily codec does not beat that baseline — in which case retraining per cadence
buys nothing and the daily arm should go back to reusing the codec.

**What this dispatch actually risks is the PREAMBLE, not the science.** #389
was cancelled at ~7 h having never written a metric line, so #400 re-runs its
EXACT inputs — recovered verbatim from `probes-389.json`'s provenance block on
`ml-metrics`, which archived provenance only and is the sole surviving record
of them, since a cancelled run's log blobs expire. Only `doc` differs. `resume`
was ABSENT from #389's provenance and is passed here as `""`; that is the
workflow's own default, so it is not a change. The fixes it is testing are
`aea7d63` (anomaly_transform time-chunked, 249.8 → 6.0 full-extent traversals,
bit-identical, peak RAM ~4.5 GB at chunk 64), `22a5b27` (the third and fourth
inlined copies of the same transform) and the probe memory work in `70ffe2d` /
`f2ee8b8`.

**A falsifier for the FIX, stated at dispatch and cheap to check.** The
`metrics.jsonl` config line must appear on `ml-live-400` within ~45 min of the
Train step starting. **If it is absent at 90 minutes — 22:11:47Z — the fix did
not take and the run must be KILLED rather than left to burn.** #389's failure
mode was a job that looked healthy for seven hours, so "still running" is not
evidence here and a deadline stated in advance is the only instrument that
works.

| | |
|---|---|
| run | [#400](https://github.com/blauewelt/earth/actions/runs/32183046877) |
| box | `gpu-box-46996216` (Vast 47913006, 700 GB disk, 64,164 MB RAM, $0.333/h) |
| code | dispatch ref `main` = `a1c7411` |
| params · batch · steps · data | 37,975,889 · 512 · 200,000 · `[15706, 281, 481, 39]` float16, 165.6 GB |
| build step | **0 seconds** (20:35:15Z → 20:35:15Z) |
| provenance step | 6 min 30 s (sha256 over the 165.6 GB sidecar) |
| Train step began | 20:41:47Z |

The 0-second build is the #391 (read-out ladder, attempt 1) guard ordering working as intended: the recipe
short-circuit sees #389's cached `f5r1` tensor and returns before the free-space
check, and `load_pentad_base` opens the daily base fields with `mmap_mode="r"`,
so confirming "already built" costs no read at all.

**THE FIX TOOK — measured, 21:52:53Z.** The `metrics.jsonl` config line appeared
on `ml-live-400` at **21:52:53Z**, i.e. **~71 minutes** after the Train step
began, inside the 90-minute deadline and later than the 20–40 min the fix's own
arithmetic suggested. It reads `d_model 512, n_layers 12, n_heads 4, d_dec 256,
d_z 32, patch 1, batch 512, steps 200000, params_M 37.976, data
family5_na025_daily.npz, C 39, T 15706, resume null` — #389's intended
configuration, on the daily tensor. `gpu_util` went to **98%** at the same
moment. **The daily arm exists**: #389 spent seven hours in
`anomaly_transform` and never reached this line.

The config line is written at `train.py:502`, *after* `anomaly_transform` at
`train.py:274`, which is exactly why its appearance is the proof and not merely
a sign of life — and why the deadline was set on it rather than on CPU or GPU
utilisation, both of which looked identical during #389's seven-hour hang (0%
GPU, ~1 core busy) and during this run's healthy 71-minute preamble.
The preamble decomposes as the ~3 min `writable_copy` #389 measured, the
time-chunked transform, and the `obs_any_chunked` / `pool_idx` pool build — the
last two not separable from outside the box, since a running job's step log is
not served.

**Disk, verified before dispatch rather than assumed.** The tensor is on the
box already, but `train.py` still takes a writable `_scratch.npy` copy of the
WHOLE thing for the anomaly transform (`tensor_io.writable_copy`, chunk 64,
peak RSS one chunk — #389 measured that copy at 2 min 45 s), so the run needs a
second 165.6 GB. 364/700 GB used → **336 GB free, ~170 GB margin**.
`disk_hygiene.sh` runs at a 16 GB floor and therefore does nothing on this box,
which is what keeps the cached tensor safe.

**Result: CANCELLED 2026-08-19 01:35Z at 22,000 / 200,000 steps** — in the
credit triage at the top of this file, not on any scientific ground. The daily
arm's economics that the 22k steps did measure (216.6 ms/step, probes 65% of
wall, the light-probe trace) are recorded there. Re-dispatch verbatim when
credit allows; the box keeps the tensor and a ~22k orphan checkpoint.

**Result (pentad trained arms): BOTH read-out ladders are now RESOLVED.** #406
(the frozen anchor's ladder, above): the frozen anchor at pooled **0.660**, unpooled head **0.691**, matched
raw-3×3 control **0.683**, wind bar **0.670** — the head clears wind, and the
anchor's embedding beats raw by +0.008, i.e. by nothing. #409 (the VERDICT
section above): #386's OWN codec through the identical head, pooled **0.652**,
head **0.680**, its own raw control **0.683** — the trained embedding beats raw
by **−0.003**, i.e. by nothing, in the other direction. No embedding at pentad
adds transport information beyond the raw fields.

---

## E-037 · xl233 × znoise: the corner that completes the factorial — QUEUED 2026-08-16 ~22:00Z

Chris, reading the wave-8 dispatch: *"I thought you proposed xl233 + znoise
above. Why xl144? Let's queue xl233 + noise after."* He is right that the
proposal was ambiguous — the message said "the 233-point stencil, plus the
noise × 205M composition", and E-036 resolved "205M" to xl144 without saying
so. **The reason for that choice was interpretability, not interest**: xl144
is the only width with a paired clean control already measured at 205M
(#346/#347 → 0.68067 / 0.67558), so the noise delta lands on one flag. xl233
clean did not exist when E-036 was written; it was starting in the same wave.
That argument is about which cell is cleanest to READ, and it is not a reason
to leave the most interesting cell unrun.

### What this arm actually buys

With it, the 205M tier becomes a complete **2×2 factorial**, which no part of
this programme has had on the rolled scoreboard:

| | clean | znoise σ=0.7 |
|---|---|---|
| **sunflower-144** | 0.6781 (#346/#347, eval #356) | #359/#360 — in flight |
| **sunflower-233** | #357/#358 — in flight | **#361/#362 — this entry** |

The three cells already running give two main effects. Only the fourth gives
the **interaction**, and the interaction is the question §8.7 of the paper
actually poses: width adds live input, noise adds robustness to corrupted
input, and §8.6 measured that ~50% of a 4444 km stencil's slots are *already*
dead zeros. If those two mechanisms are the same mechanism — if wide-and-far
inputs help precisely because they teach a model to work with a half-empty
context — the corner cell comes in BELOW the sum of its main effects. If they
are independent, it lands at the sum. Neither is predictable from the other
three cells, which is exactly what makes it worth the box.

**Hypothesis.** Sub-additive: corridor AUC above both single-intervention
cells but below (xl233 gain) + (znoise gain) added onto xl144-clean's 0.6781.

**Falsifier.** At or above the additive prediction = width and noise are
independent axes and the "dead-slot robustness" reading of §8.6 is wrong.
A cell BELOW both single interventions would say they actively conflict —
the only outcome that would make the composition a mistake rather than a
measurement.

**Honest note on sequencing.** §4.4 calls sequenced experiments a smell, and
if E-036 falsifies — noise dead at 205M — this arm was ~$10 spent on a corner
of a factorial whose interesting axis collapsed. Running it concurrently
rather than after E-036's verdict is a deliberate trade: $10 against a day of
wall-clock, on a fleet that is otherwise idle the moment wave 8 lands.

### Dispatch — QUEUED, then MOVED ONTO RESTARTED BOXES

First dispatched as **#361/#362**, deliberately queued behind E-036 on its own
two boxes. Chris then said *"let's run those now"*, so two stopped boxes were
restarted and the arms re-dispatched onto them as **#363/#364**; #361/#362
were cancelled once #363/#364 were confirmed picked up (dispatch first, verify,
then cancel — the reverse order leaves a window with no arm at all).

| run | arm | seed | runner | Vast |
|---|---|---|---|---|
| **#363** | xl233 + znoise 0.7 | 0 | gpu-box-45731106 | 47718230 |
| **#364** | xl233 + znoise 0.7 | 1 | gpu-box-30257785 | 47726876 |
| ~~#361/#362~~ | superseded, cancelled while still queued | | | |

**Vast `start` is not reliable, and the API says so quietly.** Four of five
stopped boxes answered `resources_unavailable` with *"state change queued"* —
and that queue did not hold: `intended_status` read back as `stopped` minutes
later, so the box was never coming. Only two of five actually started. Read
`actual_status` AND `intended_status` after any start; the success message is
not evidence (ml/CLAUDE.md §0.2). A fifth box (47724565) accepted the start
late and was stopped again rather than left to drift into an unplanned $0.31/h.

Restarting a stopped box was preferred over renting new ones: it keeps the
disk, the runner registration, the data cache and the checkpoint mirror, so it
is both cheaper and warmer. The standing rule that NEW boxes must be
large-disk 600 GB+ is about renting, not about restarting what we already own.

This also means the fleet is now **six arms on six boxes** — the two xl233
boxes are no longer reserved for tomorrow's eval wave, which will need a box
freed by whichever arm lands first.

```
stencil:234,ring:spiral:111-4444-0.71-0.5,seed:0,sched:expdecay --lr-cooldown-frac 0 --milestone-steps 600,60000,120000 --input-znoise 0.7
```

That also fixes the fleet allocation for tomorrow: the two xl233 boxes
(gpu-box-42005419, gpu-box-46045353) stay free for the wave-8 **eval** wave,
which needs only two `sroll:` runs — one per arm, both seeds plus the gate in
a single dispatch, the way #352 rolled six heads at once.

### HARVEST — #394 (the E-037 eval) landed 2026-08-18 ~23:00Z: noise × width is the one interaction that COMPOUNDS

**Gate PASSED:** `e017_u1_s0` reproduced `horizon_auc` **0.643 exactly** (tol
0.0101), so the wave is valid and the numbers below are readable.

| arm | corridor AUC | gate-scope AUC |
|---|---|---|
| xl233 + znoise 0.7, **seed 0** | **0.725** | **0.780** |
| xl233 + znoise 0.7, **seed 1** | **0.723** | **0.778** |
| **noised pair, mean** | **0.7240** | 0.779 |
| E-035b xl233 **clean** control | 0.673 | 0.720 |

**Noise × width compounds: +0.050 on the corridor** at the 205M / xl233 rung,
and **+0.058** in gate scope, so the effect is not an artefact of one scoring
region. The noised pair's seed spread is **0.002** — tighter than the clean
tier's own — so the delta is many times the noise it must clear. Read this
alongside E-035's finding that clean width SATURATES at 233 points: the
composition gains its 0.050 on **exactly the rung where width alone gained
nothing**, which is what makes it an interaction rather than two main effects
added together.

The mechanism (the exposure-bias horizon signature: worse at h=1, better at
every horizon after, gap widening monotonically to +0.076 at twelve months), the
width-saturation reading, and the transport caveat that survives all of it are
written up in the paper — `ml/paper/paper.tex`, new Table 6, committed
`c4c900c` — rather than duplicated here.

---

<a id="e-035"></a>
## E-035 / E-036 · The two compositions nobody has run — DISPATCHED 2026-08-16 ~18:40Z · BOTH RESOLVED 2026-08-19

**Both verdicts are below in this section.** E-036 (input noise × xl144)
resolved 2026-08-19 ~03:12Z on #401 (the E-036 eval): noise still pays at
205M, §"E-036 RESOLVED". E-035 (clean width at 233 points) resolved
2026-08-19 10:12Z on #413 (E-035 seed-0 roll-forward), which completed the
seed pair: §"HARVEST — #413" — the falsifier fired and clean width buys
nothing past 144 points.

Wave 8. Four training arms across the four warm boxes, ~200k steps each at
the 205M tier. Chris approved the spend after the #354–#356 harvest: *"sounds
great, let's start it."* Both arms exist because that harvest **reversed a
conclusion this log had already recorded**, and the reversal has a second
prediction each.

### The reversal being acted on

E-032 measured 89 vs 144 points at 205M/200k on the FORECAST ratio and closed
the width axis: paired −0.00004 / −0.00012, ~17× inside seed spread, and this
log said *"the width axis CLOSES at 89"* and *do not build 233*. #355/#356
then rolled the same two checkpoints twelve months:

| arm | s0 | s1 | mean | vs previous rung |
|---|---|---|---|---|
| xl55 (60k) | 0.664 | 0.663 | 0.6637 | — |
| xl89 (200k) | 0.67433 | 0.67058 | 0.6725 | +0.0088 |
| xl144 (200k) | 0.68067 | 0.67558 | **0.6781** | +0.0056 |

Paired by seed xl144 − xl89 = **+0.0063 / +0.0050**, same sign at both seeds,
monotone ladder, no flattening. **Saturation on a one-step metric is not
saturation under iteration.** The programme has now made that error in both
directions — E-010 closed the capacity axis in a regime that could not
exercise it, E-032 closed the width axis on a scoreboard that could not see
it. A settled negative is settled only on the scoreboard that settled it.

### E-035 · sunflower-233 × 1024×16 × 200k, seeds 0 and 1

**Hypothesis.** Corridor AUC ≈ **0.683 ± 0.005** if the +0.005/rung trend
continues — above xl144 and at or above znoise-big55's 0.6785. The forecast
ratio stays ≈0.1225, indistinguishable from xl89 and xl144: width is
genuinely finished on that axis and nothing here should move it.

**Falsifier.** xl233 within seed noise of xl144's 0.6781 = width finally
saturates on the roll too, and the ladder has a top.

**The confound, pre-registered because it is the whole risk.**
§8.6's cone measurement says ~50% of a 4444 km stencil's slots already
resolve to land or off-window. If 233 returns a null, *"width saturated"* and
*"the window ran out"* look identical from the AUC alone. So occupancy was
measured BEFORE dispatch (`ml/measure_slot_occupancy.py`, row added):

| design | slots | rolled live | corridor live |
|---|---|---|---|
| sunflower 34 | 35 | 50.2% | 47.6% |
| sunflower 55 | 56 | 50.2% | 47.4% |
| sunflower 89 | 90 | 50.0% | 47.1% |
| sunflower 144 | 145 | 49.9% | 47.0% |
| **sunflower 233** | **234** | **49.9%** | **46.9% (109.4/233)** |

Scale-invariant to a tenth of a point across a 7× range of counts. **A null at
233 therefore cannot be blamed on occupancy** — more points buy proportionally
more live input, exactly as at every rung below. If it IS a null, width and
window have become the same axis and only the global tensor (E-033/E-034) can
separate them.

**Artefact checks done at dispatch, not at use.** `build_stencil` builds the
234-slot table (the occupancy run above IS that check). A CPU toy instantiated
both heads and ran forward+backward: xl144 **211.4M** params, xl233
**217.3M** — the width axis costs only the input projection, +5.9M — all
gradients finite, `pred` shape `[B,K,64]`. Input gather at B=512 goes
0.42 → 0.69 GiB per copy, +0.27 GiB on a card that already ran 145 slots.

### E-036 · znoise σ=0.7 × sunflower-144 × 1024×16 × 200k, seeds 0 and 1

The other half of the #352 result. At 88M, σ=0.7 input noise bought **+0.057**
corridor AUC and produced a model (0.6785) that beat the clean 205M one
(0.6637). Nobody has composed it with capacity.

**Hypothesis.** Noise and scale compound: corridor AUC **> 0.6781** (xl144
clean, the paired control at identical width and capacity), plausibly ~0.72 if
the 88M delta survives intact. The forecast ratio gets WORSE, as it did at 88M
(0.1548/0.1539 vs 0.1476) — that cost is the mechanism, not a defect.

**Falsifier.** ≤ 0.6781 = input noise was a small-model regulariser whose
benefit capacity already supplies, and the +0.057 does not generalise up the
ladder. That would close the arm and leave exposure bias to E-030's mechanism.

**Control.** xl144 clean, 0.68067 / 0.67558 (#356) — same width, same
capacity, same steps, same seeds, differing in one flag.

### Why these two and not a σ sweep

σ=0.7 was set from the model's own measured one-step error and never tuned, so
a sweep is real work — but it is only worth doing if noise still pays at 205M,
which is exactly what E-036 asks. Sequenced experiments are a smell
(§4.4), and this is the rare case where the sequence is genuine: the sweep's
existence is conditional on this run's sign.

### Dispatch record

Four warm boxes, all four having just finished #354–#356, so each holds the
sha-pinned tensor, the `run-62/63` codec and a warm Z cache.

| arm | seed | runner | Vast |
|---|---|---|---|
| E-035 xl233 | 0 | gpu-box-42005419 | 47487801 |
| E-035 xl233 | 1 | gpu-box-46045353 | 47717160 |
| E-036 znoise×xl144 | 0 | gpu-box-47094143 | 47720660 |
| E-036 znoise×xl144 | 1 | gpu-box-47529389 | 47720664 |

`--milestone-steps 600,60000,120000` on every arm: the 600 rung is the
early-save proof (Chris, 08-15: *"execute the checkpoint save also early on —
otherwise we risk a crash after 60k steps"*), and the 60k/120k rungs make each
run its own step-budget ladder instead of a single terminal number.
`job_timeout 1800`, because 200k steps at this tier is ~15–18 h and
`job_timeout` is an input, not a cap.

**Cost, recorded at dispatch per §3:** 4 boxes × ~18 h × ~$0.28/h ≈ **$20**,
plus the eval wave that must follow (a `sroll:` run per arm, ~6 h each) before
any of it is a result.

### E-035 seed 0 · #396 completes the pair — 2026-08-19 06:10Z

**#396** (dispatched 2026-08-18 14:34Z on `gpu-box-45731106` / Vast 47718230,
run id 32149161274) is the second attempt at the E-035 seed-0 arm, after #395 (attempt 1)
died in 90 s carrying no codec architecture and after the ORIGINAL seed-0 head
was lost as a 604 MB fragment of a 2.6 GB checkpoint. It ran the full **200,000
stage-2 steps** on the frozen `run-62,run-63` f3 anchor codec, and its own
`stage2_config` on `ml-metrics/run-396.jsonl` records the arm exactly:
`d_model` **1024**, `layers` **16**, `K` **24**, `stencil` **234**, `ring_km`
`spiral:111,4444,0.71,0.5`, `seed` **0**, `unroll` 1, `params_M` **217.276**,
batch 256, 28,857,960 train windows. Terminal step 200,000:
`stage2_val_zmse` **0.36782**, `stage2_zmse` 0.02937, `stage2_amp` 0.9734,
`stage2_lr` 3.235e-05, wall **55,178 s** (15.3 h). The light in-training probe
(NOISY, trend only — rule 5) read `rapid_r_deseas` 0.425 → 0.401 → 0.421 →
0.415 → **0.429** across 120k–200k.

**#396 is a GREEN run with no `temporal.json`, and for once that is not the
usual story.** §7's signature says the trainer died and nothing noticed. Here
the trainer **finished**: the log prints `step 200000/200000 z-mse 0.0294
(55265s)` at 06:08:11Z and the process was **OOM-killed 31 s later** — `Killed
… exit code 137` at 06:08:42Z — inside `temporal.py`'s POST-LOOP evaluation,
before it could write its results file. So `probes-396.json` carries
`probe_sequence.json` and `provenance.json` and nothing else: there is **no
`rapid_probe_kfold` for this arm and there will not be one**, and the run went
green only because `Probes` is `continue-on-error`. Two consequences worth
naming. The head itself is INTACT, because `temporal.py` mirrors it inside the
loop on `s % log_every == 0 or s == a.steps` — i.e. the 200,000-step mirror was
written before the print that preceded the kill. And the arm's number must now
come from the ROLL (`rollout_spatial.py`), which is the primary instrument
anyway; nothing is lost that this programme quotes.

**The head existed on exactly one rented disk.** `snapshot_head.sh` warned every
~30 minutes for fifteen hours — `upload of run-396-temporal-latest.pt FAILED` —
because a 217M head with Adam moments is 2.6 GB and a release asset caps near
2 GiB. That is the wave-8 defect `publish_head_weights.sh` was written for, and
it is why the publication below had to happen on the box itself.

**Publication — #412 (headpub of `e035a-xl233-s0`), 2026-08-19 06:21Z** ([run](https://github.com/blauewelt/earth/actions/runs/32223060484)),
`window: headpub:e035a-xl233-s0@temporal`, inputs otherwise #379's verbatim
25-field block with the runner pinned to `gpu-box-45731106` (started for this
purpose from `exited`; 43 GB free, well above `disk_hygiene`'s 16 GB trigger, so
the box's warm Z cache was never at risk). Published
**`head-weights-e035a-xl233-s0.pt`**, 869,180,725 bytes, HTTP **201**.

Three details of the mechanism that cost a day on 2026-08-17 and should not be
rediscovered:

- **`@temporal` is mandatory on a box whose run COMPLETED.** The stage-2 head
  mirror lands at `/opt/earth-cache/ckpt/temporal.pt` — bare, with no run tag —
  because the `Probes` step does not set `CKPT_TAG` (only `Train` does), so
  `temporal.py`'s `tag = os.environ.get("CKPT_TAG", "")` resolves empty. The
  default `headpub:` source is `orphan-temporal-latest.pt`, which on this box was
  a **stale 1.07 GB Aug-15 leftover**. Publishing that leftover is exactly what
  #377/#378/#381/#382 did.
- **#383/#384/#385 did NOT fail at publication.** All three printed HTTP 201 and
  put correct heads on the release; `head-weights-e035b-xl233-s1.pt` is #383's.
  They went red in an EARLIER step: their dispatches left `tensor` at the
  workflow default **`family2`**, which routes the build step past the family3/4/5
  branches into `python ml/build_dataset.py --window "${WINDOW}"` — and
  `build_dataset.py --window` takes only `global|na`, so `headpub:…@temporal` is
  an `invalid choice` and the step exits 2. #379 never hit it because
  `family3_na025` takes the first branch and `build_dataset.py` is never invoked.
  The red is a `tensor`-input bug, not a publication bug; #412 carries
  `family3_na025` and went green end to end.
- **The `step` key does not survive.** `publish_head_weights.sh` keeps only
  `{args, model}`, so a weights-only asset has no `step` and no `run_number`.
  The 200,000 is corroborated twice instead: the publisher printed
  `step=200000 d_model=1024 layers=16 stencil=234 seed=0 znoise=0.0`,
  `params=217.3M` from the SOURCE, and the asset's own `args.steps` is 200000.

**Verified on the PUBLISHED BYTES, not on the publisher's log** (§0.1 — and
#382's stale-leftover is why this check is not optional). The asset was
downloaded and `torch.load`ed in the sandbox: `d_model` 1024, `layers` 16,
`K` 24, `stencil` 234, `ring_km` `spiral:111,4444,0.71,0.5`, `seed` **0**,
`input_znoise` 0.0, `steps` 200000, `lr_schedule` expdecay,
`data` `family3_na025.npz`, **217,276,480 parameters** across 199 tensors —
matching `stage2_config`'s `params_M` 217.276 to the digit. The load-bearing
shape is `inp.weight` **(1024, 14978)**: 234 slots × `d_z` 64 = 14,976, +2. An
xl144 head — the thing #382 published by mistake — would read 9,282 there.
Downloading the sibling and diffing the two `args` dicts leaves **exactly one
differing field, `seed` 0 vs 1**: a genuine seed pair, not two configurations.

**Eval dispatched as #413** (sroll of that freshly published seed-0 head), 2026-08-19 06:30Z
([run](https://github.com/blauewelt/earth/actions/runs/32223688147)), on the same
warm `gpu-box-45731106` (its `Z_actions_6c52f0687b_adcbe700fb.npy` and the
sha-pinned tensor are both resident, so the eval pays neither the 5.2 GiB pull
nor a rebuild). `window: sroll:e017_u1_s0,head-weights-e035a-xl233-s0`,
`job_timeout` 700, every other field copied verbatim from the E-036 eval **#401**'s
working block; plan published as `plan-413.json`. Two heads only.

**Harvest criteria, written at dispatch.** The `e017_u1_s0` gate must reproduce
`horizon_auc` **0.643** within `GATE_TOL` **0.0101** or the run is void;
`len(rollout_spatial.json['heads'])` must be **2** (#353 went green holding 2 of
6 after a CUDA OOM); the seed-0 corridor AUC is read against seed 1's **0.673**
(#394, the E-037 eval) and the PAIR MEAN against xl144 clean **0.6781**. **Falsifier unchanged
from E-035's dispatch:** a seed mean within seed noise of 0.6781 = clean width
SATURATES at 233 points and the ladder has a top. That reading is currently
carried by an n = 1 number, which ml/CLAUDE.md §3 says means nothing — and
E-037's "noise × width compounds" conclusion leans on it as its control, so the
pair is not bookkeeping.

**First minutes verified, 06:44Z** (§2 — measurements, not intentions).
`ml-live-413` is emitting on the 2.5-minute publisher cadence; head **1 of 2**
is `s1_s0` — the gate — so it LOADED, and it is stepping at a flat
**2.94 s/window** (20 / 40 / 60 / 80 of 714 at 59 / 118 / 176 / 235 s), i.e.
~35 min for the gate. `gpu_util` on Vast 47718230 reads **64%**, so this is on
the card and not on the CPU (§2, the four eval scripts that silently embedded on
CPU). `eta_all_s` ~4,000 s is a KNOWN UNDER-ESTIMATE and should not be quoted:
the reporter assumes "heads run at the same cost", and a 234-slot head is several
times a 1-slot gate — expect ~3.5–5.5 h total, against `job_timeout` 700 min.

**Cost:** #396 ~15.6 h × $0.2944/h ≈ **$4.6**; #412 ~5 min ≈ **$0.03**; #413
~3.5–5.5 h × $0.2944/h ≈ **$1.0–1.6**. Fleet at 06:41Z: credit **$49.85**,
3 boxes running, burn **$0.9352/h** — the status page's budget block was
projecting **$0.00 / runway 0.0 h** from a snapshot taken before the top-up, so
`scripts/publish_fleet_status.mjs` was re-run and `ml-metrics/fleet.json` now
carries the real numbers.

<a id="e-035-pair"></a>
### HARVEST — #413 (E-035 seed-0 roll-forward) landed 2026-08-19 10:12Z: the xl233 pair is COMPLETE, and clean width buys nothing past 144

**#413** · sroll of `head-weights-e035a-xl233-s0` against the pinned gate
(the seed-0 half of E-035) · params 217.3M · stage `sroll` · data
`family3_na025` · arch 1024×16 `stencil:234 ring spiral:111,4444,0.71,0.5`
`d_z` 64 · steps 0 (eval-only; the head is the 200,000-step terminal
mirror of #396, the E-035 seed-0 200k training run) · resume
`head-weights-e035a-xl233-s0` (published by **#412**, headpub of that head).

**All three harvest criteria written at dispatch PASSED.** The `e017_u1_s0`
gate reproduced `horizon_auc` **0.643 exactly** (tol 0.0101, `pass: true`,
`fails: []`), and `len(rollout_spatial.json['heads'])` is **2** — the gate and
the seed-0 head, no silent OOM drop-out of the kind #353 (the wave-7 sroll
that went green holding 2 of 6 heads) had. `probes-413.json` is on
`ml-metrics`.

**Corridor AUC, recomputed to five decimals from the twelve archived
per-horizon `msss_clim` values** (the stored `horizon_auc` field is rounded to
three, which is not enough to read a 0.002 pair delta):

| arm | run | stored | recomputed | source |
|---|---|---|---|---|
| E-035a xl233 clean, **seed 0** | #413 (E-035 s0 sroll) | 0.675 | **0.67492** | `probes-413.json` |
| E-035b xl233 clean, **seed 1** | #394 (E-037 eval) | 0.673 | **0.67292** | `probes-394.json` |
| **pair mean** | | | **0.6739** | |
| pair \|Δ\| | | | **0.0020** | |

**The falsifier fired.** E-035's dispatch predicted **0.683 ± 0.005** if the
+0.005/rung width trend continued, and pre-registered the falsifier as *"xl233
within seed noise of xl144's 0.6781 = width finally saturates on the roll too."*
The pair mean is **0.6739**, **−0.0042** against xl144 clean
(0.68067/0.67558, mean **0.6781**, #356 — the E-032 xl89/xl144 roll). Paired by seed it is **−0.0058** (s0)
and **−0.0026** (s1) — same sign at both seeds.

**How that reads under §3b, and how it does NOT.** The right claim is **"width
beyond 144 points buys nothing measurable"**, not *"width hurts"*. The −0.0042
is about twice the sd this tier's replicate record assigns to a difference of
two paired means (pooled sd **0.0021**, 7 dof) — enough to exclude the
+0.005/rung extrapolation that was the hypothesis on the table, not enough to
establish a sign, and far inside the **0.025** effect §3b requires before a
single seed per arm would have sufficed. What the pair DOES license is stating
the number as a **level** rather than a consistency: §3b's worked example
(*"❌ `xl233 rolls at 0.673` — a level, from one seed"*) is now resolved, which
is exactly the upgrade path #396 (the seed-0 retrain) and #413 (its
roll-forward) were dispatched to buy. The clean
width ladder in full: **0.6637 → 0.6725 → 0.6781 → 0.6739** across
55 → 89 → 144 → 233 points, monotone to 144 and flat-to-slightly-down after.

**§3b's own table is unchanged by this**, because it was written from these
numbers: E-035's 0.0020 is the smallest of the five xl-tier pair deltas
(0.0020, 0.0023, 0.0033, 0.0038, 0.0051) and is already inside the pooled
0.0021 (7 dof).

**Downstream: the E-037 control gets its second seed.** The
*"noise × width compounds"* reading in the #394 harvest above leaned on the
clean xl233 arm as an n = 1 control. With the pair in hand the 2×2 is complete
at both seeds and the noise effect at 233 points reads **+0.0503** (s0:
0.72517 vs 0.67492) and **+0.0500** (s1: 0.72292 vs 0.67292) — the same number
to three decimals. Against xl144's **+0.0454** the interaction term is
**+0.0047**, which is the size of the seed spread it would have to clear, so it
stays unclaimed — the paper's *"a correction, recorded"* paragraph already
retired the compounding reading and this pair does not revive it. Noise is a **main effect of constant size**
(+0.057 at 88M/55pt, +0.045 at 205M/144pt, +0.050 at 205M/233pt).

**The read-out is the roll, and that is symmetric across the pair.** Neither
seed has a stage-2 `rapid_probe_kfold`: **#396** (seed 0) was OOM-killed 31 s
after printing `step 200000/200000` inside `temporal.py`'s post-loop eval, so it
archived `probe_sequence.json` + `provenance.json` and no `temporal.json`; and
**#358** (the E-035 seed-1 200k training run) archived exactly the same two
files. Both arms are therefore
read from `rollout_spatial.py` alone — the primary instrument for this metric
anyway — so the comparison is instrument-matched and nothing is lost that this
programme quotes. Verified by reading `probes-396.json` and `probes-358.json`
on `ml-metrics`, not inferred.

**Paper updated in the same commit** (§3b: *"when a new replicate lands, the
table is extended in the same commit as the result"*): `ml/paper/paper.tex`
Table `tab:auc` carries the xl233 row at 0.675/0.673/**0.6739**, the `n = 1`
hedge in its caption is retired, the 233-point-rung paragraph is rewritten as a
pair, `tab:twobytwo` is now two seeds in every cell (clean-233 0.6739,
Δ noise +0.0501, Δ width −0.0042, interaction +0.0047), and `tab:noisehorizon`
now compares two-seed mean against two-seed mean (h=1 −0.006 → h=12 +0.074,
max +0.077 at h=11).

**One bookkeeping inconsistency found while verifying, NOT fixed here** (it
belongs to E-036/E-037, not E-035, and moving it would change published
deltas): the paper's two **noised** 205M means are averages of the 3-decimal
`horizon_auc` field, while every clean mean in the same table is recomputed
from the per-horizon arrays. znoise×xl144 recomputes to **0.723705**
(0.72208/0.72533, #401 — the E-036 znoise×xl144 eval) i.e. **0.7237**, not
the 0.7235 the table prints; znoise×xl233 recomputes to **0.724045** i.e.
**0.7240**, which matches. On the
recomputed convention the two noised rows sit **0.0003** apart rather than
0.0005, Δ-noise at 144 points is **+0.0456** rather than +0.0454, and the
interaction term is **+0.0045** rather than +0.0047. None of those move a
conclusion — every one of them is an order of magnitude inside the tier's
spread — but the table should pick one convention.

**Cost of the pair, for the §3b ledger:** #396 15.6 h ≈ **$4.6**, #412 ~5 min
≈ **$0.03**, #413 13,005 s of head wall (1,986 s gate + 11,019 s xl233) ≈ 3.6 h
≈ **$1.1** — about **$5.7** to turn a consistency into a level, against §3b's
estimate of "about $6 and about a day of a rented box".

### E-036 eval · #393 died with nothing archived — re-dispatched as #401

**#393** (2026-08-18 13:29Z, `gpu-box-42005419` / Vast 47487801,
`sroll:e017_u1_s0,head-weights-e036a-zn-xl144-s0,head-weights-e036b-zn-xl144-s1`)
ran 4 h 42 m and returned **no number**. At **18:11:25Z the self-hosted runner
lost communication with GitHub**: the job closed as `failure`, its log blobs
were deleted, and **no `probes-393.json` exists on `ml-metrics`**. The last
`ml-live-393` push (17:59:59Z) shows head 1 (the gate, `s1_s0`) and head 2
(`s145rspiral:111-4444-0.71-0.5_s0`) through all three sroll phases and head 3
(`…_s1`) at 28% of `skill` — but `ml/rollout_spatial.py` writes
`rollout_spatial.json` only at the END, so **two completed rolls died with the
runner**. §5.20 (publish a shared artefact when it EXISTS, not when the job
ends) is the rule this evaluator does not follow; that is an open lever, not
something changed here. The box is UA-hosted, is left STOPPED, and is treated
as suspect.

**Re-dispatched 20:37Z as [#401](https://github.com/blauewelt/earth/actions/runs/32183242672)
on `gpu-box-31479844`** (Vast 47724559, reliability 0.994, 58 GB free,
$0.2944/h). The inputs had to be RECONSTRUCTED — #393's provenance was never
archived and its log is gone — and every field is corroborated by at least two
sources: `ml-live-393`'s head labels, which `rollout_spatial.py:795` derives
from each head file's own `args` and which therefore identify the ARTEFACTS
rather than the tags; the release asset names those labels can only match; the
headpub provenance that produced them (**#379** → `head-weights-e036a-zn-xl144-s0`
from #359, **#380** → `head-weights-e036b-zn-xl144-s1` from #360, each run on
the very box that trained its arm); and the 25-field `sroll:` dispatch that is
identical across #355, #356 and #382. Sibling **#394** (E-037 eval, same wave,
still healthy past 7 h) puts `job_timeout` above the 350-minute default,
confirming the 700 the two earlier sroll runs used. `ml/rollout_spatial.py`,
`scripts/sroll_run.sh` and every module they import are **byte-identical**
between #393's sha (`bfd2248`) and #401's (`9dcbd65`), so the two runs measure
the same thing and both are comparable to #356's 0.6781.

Harvest criteria unchanged: the `e017_u1_s0` gate must reproduce **0.643**
within `GATE_TOL` = 0.0101, `len(rollout_spatial.json['heads'])` must be **3**
(#353 went green holding 2 of 6 after a CUDA OOM), and the two znoise xl144
corridor AUCs are read against xl144 clean **0.6781**. Falsifier unchanged:
**≤ 0.6781** = input noise was a small-model regulariser whose benefit capacity
already supplies.

**Cost:** #393 burned ~4.7 h × ~$0.26/h ≈ **$1.2** for nothing; #401 is
~6.5 h × $0.2944/h ≈ **$1.9**.

**One signature worth not misreading.** #393 and #394 both had their `Train`
step FAIL at exactly 2 m 30 s and continued into a healthy sroll regardless —
`Probes (K-sweep + stage 2)` carries `if: always()` and `continue-on-error:
true`, so the eval never depended on it. That failure was a real regression at
`bfd2248` and is now GONE: at `9dcbd65` #401's `Train` succeeded in 90 s,
because 86c012b resolves `--resume` and adopts the checkpoint's architecture
before the model is built. On an `sroll:` run a failed `Train` is not the eval
dying; read the Probes step.

### E-036 RESOLVED, 2026-08-19 ~03:12Z — #401: input noise still pays at 205M

**Run #401** (`head_sha` `9dcbd65`, archived as `probes-401.json`,
`gpu-box-31479844`). **Both harvest criteria met:** the `e017_u1_s0` gate
reproduced `horizon_auc` **0.643 exactly** (`pass: true`, tol 0.0101, bands
0.470 / 0.375 / 0.492 identical to the reference), and
`len(rollout_spatial.json['heads'])` is **3**. The wave is valid and the numbers
below are readable.

| arm | corridor AUC | gate-scope AUC |
|---|---|---|
| znoise σ=0.7 × xl144 @ 205M, **seed 0** | **0.722** | **0.780** |
| znoise σ=0.7 × xl144 @ 205M, **seed 1** | **0.725** | **0.777** |
| **xl144 clean control** (#346/#347, eval #356) | **0.6781** | — |

**The falsifier did NOT trigger.** It required **≤ 0.6781** — "input noise was a
small-model regulariser whose benefit capacity already supplies". Both seeds
land **+0.045** above it on the corridor, with a seed spread of 0.003, so the
delta is more than an order of magnitude larger than the noise it must clear.
Input noise still pays at 205 M on the 144-point stencil; the +0.057 measured at
88 M generalises up the capacity ladder rather than being absorbed by it.

**It agrees with E-037's independent pair.** The 233-stencil noised arms
(#363/#364, eval #394 — the E-037 eval) read **0.725 / 0.723** corridor and **0.780 / 0.778**
gate-scope — the same numbers to within seed spread on a different stencil and a
different eval run. Two widths, four seeds, two evaluators: the noise effect at
205 M is the best-replicated result on the rolled scoreboard. It also closes the
2×2 factorial's last unread cell, and the conditional σ sweep §"Why these two
and not a σ sweep" made contingent on this sign is now UNBLOCKED.

**Operational note — this one result cost FOUR dispatches**, and none of the
three failures were scientific:

| run (all four are E-036 eval attempts) | what killed it |
|---|---|
| **#393** | box died at 4 h 42 m with heads 1–2 rolled but UNARCHIVED — `rollout_spatial.py` writes its JSON only at the END (§5.20) |
| **#402** | no PyYAML on the box in the Resolve recipe step |
| **#403** | `TENSOR` assigned but never exported, so the step below it never saw it |
| **#401** | clean |

Two of the three are the same shape as every other loss this week: a
precondition that is free to check at dispatch and expensive to discover at use
(§0 rule 3). The third (#393) is an artefact-publication gap, not a guard gap —
an evaluator that writes only at the end converts any box failure into total
loss of everything it computed, which is the open lever §5.20 names.

---

## E-032 · sunflower-144: the width ladder's next rung — DISPATCHED 2026-08-15 ~18:05Z

**Chris (18:00Z): "Can we also add more input points? That is: the next
Fibonacci number after 89? … in a separate experiment."** Separate from
E-031 by design: the width ladder stays at the 768×12 tier where its three
existing rungs live, so the paired comparison is clean (34→55→89 each paid
−0.002..−0.003; sun89's fresh 60k baseline is 0.14556/0.14420).

**Pre-registered occupancy** (measure_slot_occupancy, row added):
sunflower-144 at r_max 4444 = **47.0% corridor live (67.7/144)** — the
same scale-invariant fraction as 34 (47.6%), 55 (47.4%), 89 (47.1%). More
points = proportionally more live input; nothing about the geometry
degrades at this count.

**REVISED 18:15Z (Chris): "the 144 points XL experiments should also run
for 200k steps (please cancel the 60k ones) … let's setup 200k and make
sure to retain some checkpoints 60k, 120k."** The big-tier 60k arms
(#338/#339) were cancelled unstarted; E-032 is now **xl144 × 200k ×2**
(1024×16, stencil:145, ~215M with the 145-slot proj). The 60k/120k rungs
are retained by the new `--milestone-steps 60000,120000` (weights-only
in-run saves riding the probes artifact, commit 7dc6fca) — the confident
single-run design, chosen over legs because an xl-tier full head cannot
cross a leg boundary (>2 GiB, no release seed; today's green-dead lesson).
#336/#337 (xl89, 20 min in) were also cancelled and re-dispatched with
milestones so both flagship cells retain their rungs. Two more parked
boxes started for xl144; all four XL runs land ~12:00–13:00Z 08-16.

**Hypothesis**: another paired ~−0.002 vs sun89 (at xl: vs xl89's own
milestones, paired at every rung). **Falsifier**: within seed noise of xl89's milestone at the matched step
= width saturates under the large head.

## E-031/E-032 HARVEST (#344–#347, 2026-08-16 ~10:30Z) — the stack lands, and the WIDTH AXIS CLOSES at 89

All four 200k XL arms green with temporal.json. **New project bests on the
forecast axis, and the first clean saturation anywhere in the scale
programme.**

| arm | stencil | params | s0 | s1 | mean |
|---|---|---|---|---|---|
| **xl89** (E-031) | 90 | 207.7M | 0.12362 | **0.12158** | **0.12260** |
| xl144 (E-032) | 145 | 211.4M | 0.12358 | 0.12146 | 0.12252 |

**(1) The three axes DO stack.** xl89-200k at 0.1226 vs xl55-60k's 0.1313
(−0.0087) and sun89-big-200k's 0.1361 (−0.0135): capacity + width + steps
compose, and the combination is the best forecaster the project has ever
had — 0.1226 against a persistence baseline of 3.126 z-units², i.e. the
one-step error is 12.3% of no-change.

**(2) The width axis is CLOSED at 89 points.** xl144 − xl89 paired:
**−0.00004 / −0.00012** — two seeds, same sign, and ~17× SMALLER than the
seed spread (0.0020–0.0021). The Fibonacci ladder paid −0.002 per rung at
34→55→89 (768×12); at 205M the 89→144 rung pays essentially exactly zero.
Falsifier for E-032's hypothesis ("ratio < xl89's") is met in sign but the
effect is inside noise by any reading, so the honest statement is
**saturation, not a win**: 144 points costs 3.7M extra parameters and ~8%
more wall-clock for nothing measurable. Do not commission a 233-point rung.
Note this is width saturating at FIXED capacity — the corridor AUC could
still separate them, and both arms' heads are published for that eval.

**(3) Chris's milestone requirement is VERIFIED END-TO-END.** Every one of
the four artifacts contains all three rungs — `temporal_ms600.pt`,
`temporal_ms60000.pt`, `temporal_ms120000.pt` — written at the named steps
with the correct step/seed/stencil in their own args, and the step-600 proof
rung fired 25 minutes into a 15-hour run, exactly as designed. A crash after
60k would have cost nothing. All twelve milestone heads published
weights-only as `e031xlx{600,60000,120000}_u1_s{0,1}` and
`e032xlx{...}`; the four finals published per INFRASTRUCTURE §2b (weights-only
asset + `.full.part00/01` + `.sha256` manifest, since a 205M head with
optimiser state is ~2.5 GB and the release cap is 2 GiB).

**Head-probe k-folds** (secondary, ~9 effective DOF — quoted as alive, not
argued from): xl89 s0 0.467 [0.367, 0.552], s1 0.470 [0.370, 0.558];
xl144 s1 0.429 [0.294, 0.544].

**Cost**: 4 arms × ~15 h × $0.26 ≈ **$15.6**. Wall 14.7–14.9 h for 200k
steps at 205M — ~265 ms/step, in family with the xl55 pace.

**Sandbox lore added (disk):** an XL probes artifact is now ~4.6 GB of zip
holding four ~830 MB checkpoints; extracting it whole needs ~9 GB and blew
the sandbox disk on the first attempt. `/tmp/harvest_xl.mjs` extracts ONE
member at a time and deletes the zip before splitting the final head (split
writes a second full copy) — peak usage is then zip + one member, or head +
its parts, never both.

## E-031 · xl89: widest input × largest transformer × 200k — DISPATCHED 2026-08-15 ~17:55Z

**Chris (17:50Z): "so the sun89 experiments are not xl? Then let's combine
89 + xl and train for 200k steps overnight?"** Correct observation — every
sun89 arm so far is 768×12. This is the flagship cell: 1024×16 (≈206M with
the 90-slot input proj) × sunflower-89 × 200k expdecay, seeds 0–1, from
scratch.

**Hypothesis.** The two axes have been additive everywhere measured
(E-027: no width×capacity interaction; #304: both transfer to the roll).
Stacking the measured effects: xl55 0.1295–0.1331 plus the big-tier width
gain (−0.0027 at 55→89) predicts ~0.127–0.130 at 60k-equivalent, and the
xl val slope (−0.008/10k at 60k, unconverged) suggests meaningfully lower
by 200k. **Falsifier**: ratio ≥ xl55's own (width stops paying under the
larger head — the interaction that was dead at 768×12 reappearing at
1024×16). The decisive number remains the corridor AUC eval.

**Mechanics.** ~260–290 ms/step expected (xl55 ran 242; the 90-slot input
proj adds a little) → 200k ≈ 15–16 h + probe ladder, so `job_timeout`
raised to 1600 min (the 350-min default and the 700-min template would
both kill it — sized against its own timeout, §1). Mid-run release
snapshots WILL silently fail (head ~2.6 GB > 2 GiB, known gap until the
snapshot-split lands with the workflow refactor): overnight insurance is
the box mirror; the completed artifact gets the split-backup treatment on
landing. Boxes: parked instances restarted if available, else queued
behind the sun89-big leg-2 runs (~20:30Z start, landing midday).

## E-030 · One-hop unroll for wide stencils (--unroll-wide 2) — DISPATCHED 2026-08-15 ~12:10Z

**Why, from Chris** (2026-08-15): *"For U equals two or three or four:
don't we need to just predict the inputs to a given pixel? So not all
pixels, and not all pixels that are four thousand kilometers away — just
the ones that are the input to the next stage."* That dependency-cone
observation makes unrolled training implementable at stencil>1, where
E-029c found plain --unroll architecturally impossible: the centre
pixel's t+1 input window needs its NEIGHBOURS' t+1 embeddings, and each
of those is exactly a depth-1 prediction from that neighbour's own
observed window — no feedback, reach-independent, S no-grad forwards per
unrolled window whatever the ring radius. Implemented as
`--unroll-wide 2` (commit 8854056: detached depth-1 pass over the S slot
pixels, dead slots zeroed — zero IS the dead-slot encoding — assembled
t+1 window, differentiable second step scored against Z[t+2] at weight
1/2, on a --uw-batch 64 sub-batch of each 512-window step; alignment
pinned by tests/test_e030_unroll_wide.py).

**Hypothesis.** Training through one hop of self-generated wide context
closes part of the train/roll input gap and improves rolled corridor AUC
over clean big55; the forecast ratio reads slightly worse (the objective
now spends gradient on a harder task). **Falsifier**: corridor AUC ≤
clean big55's (0.6213 mean, #304). Discriminates against E-029b (znoise)
— same target, different mechanism (noise is isotropic; one-hop context
carries the model's actual error structure).

**Arms**: 2 seeds on the big55 config (768×12, sunflower-55), 60k
expdecay, dispatched to the two drained boxes (gpu-box-47094145,
gpu-box-31479844). Wall estimate 5–7h/arm (the uw term costs ~55/512·S
extra forward per step ≈ 2–3× step time). Verdict comes from the E-029/
E-030 AUC eval wave.

**In-flight correction (13:50Z): the wall estimate was wrong — measured
pace is ~1.2 s/step (#322/#323), 8.5× the clean big55 step, not 2–3×.**
The dominant cost is the depth-1 GATHER, not the forward: 64 windows × 55
slot pixels × 24 months × 3,520 floats ≈ 1.2 GB re-gathered from the Z
memmap every step. 60k steps ≈ 20 h ≈ $6/arm. Left running — the term is
healthy (stage2_loss_unroll_wide ≈ 0.73–0.76 and falling, val decreasing)
and the question merits two seeds — but the next uw iteration should cut
gather cost (cache slot windows across steps, or sample S) before any
scale-up.

## THE DEPENDENCY CONE — measured 2026-08-16, and it leaves the window

**Chris:** *"are we certain we are rolling forward all the necessary pixels?
each point points towards 144 other points. Which means we need to roll
forward the whole world?"*

Measured with `ml/measure_cone_escape.py` (pure geometry over the real
`build_stencil` table, no GPU), seeding from the RAPID section:

| h | cone size | % of window | of THIS step's requested neighbours, % land/off-window |
|---|---|---|---|
| 1 | 22,738 px | 26.9% | 29.8% |
| 2 | 83,961 px | 99.5% | 38.8% |
| **3** | **84,405 px** | **100.0%** | 49.6% |
| 4–12 | 84,405 px | 100.0% | 49.8% |

*(sunflower-144, `spiral:111,4444,0.71,0.5`. For contrast the base-scale
champion ring-8@222 reaches only 30.7% of the window by h=12 with 4.8%
unmet — reach, not slot count, is what escapes.)*

**Three findings, in order of how much they matter.**

**(1) The eval already rolls everything it can.** `rollout_spatial.py`
advances all 84,405 window ocean pixels every step — that is the maximum
available, and it is why this evaluator exists. So the answer to "are we
rolling the necessary pixels" is yes *within the window*.

**(2) But the window is not big enough, and Chris's intuition is right.**
The cone saturates the entire window at **h=3**, and a 4444 km stencil in a
basin-sized window has **~50% of its slots resolving to land-or-outside at
every step** — half the input of our best model is the dead-slot encoding.
Since the corridor AUC averages h=1..12, ten of its twelve horizons are
scored on a state whose dependency cone has already left the domain. The
strict reading of the geometry is that a 4444 km reach rolled 12 months
needs 53,000 km of halo, against Earth's 40,075 km circumference: the
honest cone is the whole planet, several times over.

**(3) It is not a BUG, and the distinction is the important part.** Dead
slots are encoded as zero in TRAINING exactly as in evaluation, so there is
no train/eval mismatch and nothing here invalidates a published number. What
it means is that every rolled number in this programme is measured under a
specific, previously unstated boundary condition: **the world outside the
window is held at its climatological mean.** The corridor AUC is a
well-defined quantity; it is just not the quantity a global model would
produce, and the difference grows with reach — which is exactly the axis
E-026 was comparing when it found its inversion. (E-026b's dead-slot
measurement argues the boundary is not that inversion's mechanism —
fully-LIVE pixels carried the larger penalty, corr +0.176 — so this does not
overturn it. But it does mean reach and boundary-dependence were confounded
in that comparison, and nobody had noticed.)

**Consequences, now recorded rather than deferred:**

- The paper must state the boundary condition wherever a rolled number
  appears. Added to the limitations.
- **This promotes the global tensor from "more independent samples" to a
  CORRECTNESS requirement** for wide-stencil long rolls — E-033's Phase 4
  argument is now much stronger than when it was written this morning.
- A cheap interim experiment falls out: at fixed slot count, a *narrower
  reach* is far more self-contained (ring-8@222 leaves 95% of its demand
  met at h=12). Since the width axis just saturated at 89 points, the open
  question is whether reach can be cut without losing the capacity-era gains
  — which would buy correctness and speed at once.
- The measurement is cheap and should be run for any future stencil before
  it is trained, alongside `measure_slot_occupancy.py`.

## EVAL wave 6 · E-029 heads on the roll — DISPATCHED 2026-08-16 ~05:45Z

### WAVE 6A RESULT (#352, landed ~13:00Z) — BOTH HYPOTHESES RESOLVED, AND THE PROGRAMME'S DIRECTION CHANGES

Gate PASSED (e017 → 0.643 exactly; corridor 0.589 as always). Rolled
corridor AUC:

| arm | s0 | s1 | s2 | mean | vs clean big55 (0.6213) |
|---|---|---|---|---|---|
| **znoise-big55** (σ=0.7 input noise) | 0.683 | 0.674 | — | **0.6785** | **+0.057** |
| **ring222-big** (8 pts @222 km) | 0.646 | 0.661 | 0.661 | **0.6560** | **+0.035** |

**(1) The znoise hypothesis is CONFIRMED, and it is the largest single
effect in the roll programme.** Input noise at the model's own one-step
error scale buys **+0.057** corridor AUC while *costing* one-step forecast
(0.1548/0.1539 vs clean 0.1476). That is exposure-bias mitigation behaving
exactly as the theory says it should: trade the metric you are not scored on
for the one you are.

**THE HEADLINE, and it reorders everything: znoise at 88M (0.6785) BEATS
xl55 at 205M (0.6637) by +0.015.** A 2.3× parameter increase bought +0.042
on the roll; adding noise to the inputs of the *smaller* model bought
+0.057. **The cheapest thing we have ever done outperforms the most
expensive.** Every scale conclusion in the paper stands — capacity does pay
— but "scale is what pays most" is no longer true on the rolled axis, and
the next scale push must carry znoise rather than race it.

**(2) The ring222 hypothesis is FALSIFIED, cleanly.** I predicted the
base-scale champion's geometry would collapse into big55's band once
capacity was adequate ("geometry stops mattering"). It does the opposite:
+0.035 over big55 at the same 88M, and +0.052 over its own base-scale self
(0.6043). **Geometry SURVIVES capacity.** Eight points at 222 km beat 55
sunflower points reaching 4444 km on the roll, at equal parameters — while
losing on one-step forecast (0.152 vs 0.148). The forecast and roll axes
disagree again, in the same direction E-022 first found, and this time at
adequate capacity where the starvation explanation is unavailable.

**(3) Both winners are near-field or noise-regularised, and that is
probably one finding rather than two.** The dependency-cone measurement
earlier today showed a 4444 km stencil has ~50% dead slots and a cone
covering the whole window by h=3, where ring-8@222 stays at 4.8% unmet and
30.7% of the window at h=12. Wide-and-far models therefore roll on a state
that is half boundary assumption; znoise makes a model robust to its own
error; ring222 avoids depending on the far field at all. Both may be
attacking the same weakness from opposite ends.

**Updated corridor-AUC standings (all gate-passed):**

| rank | arm | params | AUC |
|---|---|---|---|
| 1 | **znoise-big55** | 88M | **0.6785** |
| 2 | xl55 | 205M | 0.6637 |
| 3 | ring222-big | 88M | 0.6560 |
| 4 | big34 | 87M | 0.6237 |
| 5 | big55 | 88M | 0.6213 |
| 6 | ring-8@222 (base) | 34M | 0.6043 |
| 7 | e017 gate | 34M | 0.5890 |
| 8 | base55 | 34M | 0.5710 |

**What this makes obvious as the next experiments** (none dispatched):
znoise × xl (205M) — does the largest model plus noise compound to ~0.72?
znoise × ring222 — do the two winners compose? And a σ sweep, since 0.7 was
picked from the model's own error and never tuned.

## EVAL wave 6 dispatch record

**WAVE 6B PARTIAL RESULT + AN OOM (#353, 07:45Z): WIDTH TRANSFERS TO THE
ROLL, and the evaluator has a width-scaled memory bug.** Gate PASSED
(e017 → 0.643 exactly, corridor 0.589 as always). Then:

| head | corridor AUC | vs big55 (0.6213) |
|---|---|---|
| **sun89-big 60k s0** | **0.636** | **+0.015** |

That is hypothesis 6B(a) confirmed at n=1: 89 sunflower points beat 55 on
the ROLL as well as on the forecast, and sun89-60k also beats big34's
0.6237. Width is not a one-step-only axis. (Its window AUC 0.664, gate-scope
0.689, amp h12 0.751; long-block r_heldout 0.401, lp18 0.842.)

**Then head 3 of 6 died: `torch.OutOfMemoryError: Tried to allocate
4.22 GiB` in `roll_step`'s stencil gather** — after two heads of the SAME
90-slot width had rolled fine, which is the signature of a marginal
allocation failing on fragmentation rather than a hard limit. Root cause,
and it is the third instance of one family: **`--chunk` counts PIXELS, but
the gather it bounds is [n, S, K, dz], so its true size scales with STENCIL
WIDTH.** At the 8192 default a 90-slot head requests
8192·90·24·64·4 B = 4.5 GB in a single allocation. E-027's incidents 1 and 2
taught exactly this for the TRAINER's eval batch (fixed by
`_chunked_forward`); the rollout evaluator never learned it because no
90-slot head had been rolled before.

**Fixed properly rather than by lowering a number:** `roll_step` now derives
its row count from a ~1 GiB BYTE budget and treats `--chunk` as an upper
bound — so any future width is bounded automatically (S=90 → 1941 rows,
S=145 → 1205, S=233 → 806) while narrow stencils keep the full 8192 and lose
no speed. Pinned by `tests/test_roll_chunk_budget.py`, which asserts the
budget holds at S=56/90/145/233, that the cap is never RAISED, that every
pixel is still visited exactly once, and that the stencil-1 path is
unchanged. Note the run still went GREEN with a partial
`rollout_spatial.json` — the archive's contents are the truth, not the
run's colour (§ failure signatures), which is how a 2-of-6 wave could have
been read as complete.

**Re-dispatch**: the four unscored 6B heads (sun89 s1/s2 60k, sun89x200
s1/s2) plus wave 6A's remainder if #352 hits the same wall at 56 slots.

Question on record before the numbers. #333 settled the capacity axis on
the roll (xl55 corridor 0.664, +0.042 over big). Wave 6 asks whether the
OTHER two forecast-axis results transfer, and whether the exposure-bias
mitigation does anything for the thing it exists for. Two boxes, six heads
each, gate first on both (must reproduce 0.643 or the wave is void).

**Wave 6A (mechanism at 768×12)** — ring222-big s0/s1/s2 + znoise-big55
s0/s1. Hypotheses: (a) ring-222's geometry, the BASE-scale champion, sits
WITHIN seed noise of big55's 0.6213 at adequate capacity — E-026's
inversion was starvation, so geometry should stop mattering once width is
adequate; (b) input-znoise (0.7), which COSTS one-step forecast
(0.1548/0.1539 vs big55's 0.1476), BUYS rolled corridor AUC > 0.6213 —
robustness under self-rolled inputs is its entire purpose. Falsifier for
(b): znoise ≤ big55 on the roll = input noise buys nothing anywhere, close
the arm and leave exposure bias to the E-030 mechanism. Falsifier for (a):
ring222 > big55 by more than pooled seed sd = geometry survives capacity,
E-026's ranking was real after all.

**Wave 6B (width and steps on the roll, 88M)** — sun89-big 60k s0/s1/s2 +
sun89-big 200k s1/s2. Hypotheses: (a) width transfers: sun89-60k corridor
> big55's 0.6213, as capacity did; (b) steps transfer: sun89x200 >
sun89-60k paired by seed. Falsifiers: (a) sun89 ≤ big55 = the 89-point
forecast win is one-step-only; (b) x200 ≤ 60k = the −0.009 forecast gain
from 200k never reaches the roll. Either falsification would put the
forecast ratio and the corridor AUC in open disagreement at 88M — which
E-022 proved possible and #304 made the central question of the programme.

Heads verified on the release before dispatch (all 11 assets present,
sun89x200 published full 00:50Z). Boxes: wave A on gpu-box-47094143
(Vast 47720660), wave B on gpu-box-47529389 (47720664) — both restarted
from stop with warm tensor+Z caches, both freshly drained of their sun89
legs. Est. ~6–8 h each (gate ~12 min + 5 big heads; #333's 8.7 h was
three 205M heads chunked).

## E-029d-ext · sunflower-89 → 200k steps — DISPATCHED 2026-08-15 ~14:45Z

**200k HARVEST (#350/#351, 2026-08-16 ~00:50Z) — the curve is complete and
it is bending hard.** Ratios s1 **0.13663** / s2 **0.13550** (mean 0.13607).
The full trajectory per seed: s1 0.14556 → 0.13842 → 0.13663, s2 0.14420 →
0.13689 → 0.13550. The first extension (60k steps) bought a paired −0.0072;
the second (80k steps) bought −0.0016 — per-step value down ~6×, so the
88M/89-point configuration is close to its asymptote somewhere near ~0.135.
The decisive comparison: sun89-big at 200k still does NOT reach xl55 at
60k (0.1295–0.1331) — 2.3× parameters at 1/3.3 the steps beats 1× parameters
at full budget, i.e. **at this scale parameters buy more than steps**, which
is the same verdict the forecast leaderboard has given at every rung. Both
finals verified (step 200000, expdecay, stencil 90 / spiral ring, opt
present, 89.7M params) and published FULL as
e029dsun89x200_u1_s1/s2__temporal.pt (backup + any future resume in one).
Cost of the two 120k→200k legs ≈ $7 combined. Remaining open arm: seed 0
(60k head published; continuation never commissioned — decide at the eval
wave whether n=2 on the trajectory suffices, it almost certainly does).

**UPDATE 16:10Z — first attempt GREEN-DEAD ×3, root-caused, re-dispatched.**
#332 (sun89 s1 cont) and #326/#327 (xl conts) all completed "success" in
minutes with NO temporal.json: `--resume-temporal: no checkpoint at
/opt/earth-cache/ckpt/run-<n>-temporal.pt` — refused correctly, loudly,
inside a backgrounded step, so the runs went green (the §2f/#196 signature,
now produced by design rather than a sick GPU). ROOT CAUSE: `CKPT_TAG` is
set only on the codec/joint steps, so the stage-2 mirror is the UNTAGGED
`temporal.pt`, overwritten by every subsequent job on the box and rescued
under rotating orphan names — my `run-<n>-temporal.pt` paths never existed.
Resume-by-box-path is fragile by design; **resume-by-published-name is the
fix**: the 60k heads are published FULL (opt+sched+rng, 1.076 GB, verified)
as `e029dsun89_u1_s1/s2__temporal.pt` — Chris's 60k backups and the resume
sources in one — and the continuations re-dispatched as #334/#335 via the
composable `resume2:` (workflow fix 8d5e711 doing exactly what it was built
for). The xl continuations need their 2.5 GB heads REASSEMBLED from the
split backups (the boxes no longer hold them — mirror overwritten, orphans
rotated; the split backup is now the ONLY copy besides 30-day artifacts) —
deferred to the workflow-refactor pass, which gives the seed step a script
body with room for multi-part assembly.

**E-030 arms CANCELLED BY OPERATOR 22:22:40/47Z (#322/#323, ~10h into
~21h)** — 8 s apart, deliberate; not the monitor, not any watch, no
accompanying commit. #323 had reached ~step 27k+ with in-training
rapid_r_deseas 0.509@24k and both were pacing healthily (1.19–1.29
s/step). State is PRESERVED: the box mirrors (untagged temporal.pt on the
now-parked 47717171/47724559, opt+sched+rng at ~step 27–35k) survive the
stop, so E-030 is resumable same-box for ~$3/arm (~10h) whenever wanted —
or stays closed as "gather cost made it uneconomical at this
implementation," with the code and alignment tests still landed for a
cheaper rewrite (cache slot windows across steps). DECISION PENDING
CHRIS. Cost so far ~$7.3 for zero completed arms — recorded per §3.

**sun89 s0 60k HARVEST (#325, HK box, ~7h): ratio 0.14624** — third seed
lands right in the s1/s2 band (0.14556/0.14420, mean now 0.1453); the
sun89>big55 width result holds at n=3. rapid_r_kfold 0.543 [0.422,
0.649]. Full head published (e029dsun89_u1_s0__temporal.pt).

**120k HARVEST (#334/#335, 2026-08-15 ~19:40Z): the second 60k steps buy
a paired −0.007 — step budget is a REAL third axis.** Ratios s1 0.13842 /
s2 0.13689 (from 0.14556/0.14420 at 60k), continuations verified (resumed
at exactly 60000, lr 3.58e-4, seams clean on the status page). sun89-big
at 120k now sits between xl55-60k (0.1295–0.1331) and big55-60k (0.1476):
doubling steps at 88M bought about half of what 2.3× parameters bought.
120k full heads published as e029dsun89x120_u1_s1/s2__temporal.pt (the
retained rung + leg-2 resume source). Leg 2 (120k→200k) dispatched from
them.

**60k HARVEST (with #318/#319/#321/#324 landing): sunflower-89 BEATS big55
— width still pays at 89 points.** sun89 s1 0.14556 / s2 0.14420 (mean
0.1449) vs big55's 0.1476: paired −0.0027, same size as the 34→55 width
effect (−0.0021). The width axis has NOT saturated. Also completed:
ring222-big s0 (#324) 0.15571 → trio 0.15571/0.15278/0.15088 (mean 0.1531,
sd 0.0024); znoise s1 (#321) 0.15387 → pair 0.15480/0.15387. Forecast
leaderboard: xl55 0.1295–0.1331 > sun89 0.1449 > big55 0.1476 > big34
0.1498 > ring222-big 0.1531 > znoise-big55 0.1543.


**Chris (14:35Z): "Let's make sure to immediately continue the large input
(89 points sunflower) run to 200k steps. Let's backup the intermediate 60k
and 120k checkpoints as well."** Live curves motivated it: at step ~50k the
sun89 arms read val 0.458/0.463 — under big55's FINAL 0.4615 — so width may
not saturate at 55, and the widest input deserves the longest budget.

**Structure: two legs per seed, 60k→120k (this dispatch) then 120k→200k**,
because expdecay is horizon-free (a leg boundary changes nothing about the
trajectory) and each leg's end gives a durable full checkpoint (88M+opt ≈
1.07 GB — fits the 2 GiB release cap whole) plus a probe-ladder reading, so
the ratio-vs-steps curve gets points at 60k/120k/200k for free. Backups per
Chris: the 60k heads publish as e029dsun89_u1_s1/s2__temporal.pt (full,
resumable — backup AND resume source in one), 120k as e029dsun89x120_*.

**Mechanics.** The resume2: window token now composes with stencil fields
(workflow fix 8d5e711 — tag ends at first comma; the PAT turned out to
carry the Workflows permission). But the ~30-min release snapshots have
been SILENT since ~10:59Z (run-318/319-temporal-latest.pt never appeared —
run-313/320's stopped mid-run too; watch item, cause unknown), so leg 1
seed 1 rides the sched: tail against the home-box mirror
(/opt/earth-cache/ckpt/run-318-temporal.pt, exact 60k) on gpu-box-45731106
— #328 (xl eval) cancelled and re-queued BEHIND it there, xl verdict slips
~4h — and seed 2 dispatches at ~15:45Z as resume2:e029dsun89_u1_s2 from
the published 60k head, pinned behind #324 on gpu-box-47529389. Seed 0
(#325, HK box) continuation decided when it lands.

**Hypothesis**: the 60k→120k leg buys ≥ the xl continuation's expected
gain (the sun89 val slope at 60k mirrors xl's — neither is converged);
falsifier: 120k ratio within seed noise of 60k = step-budget saturated at
this width, don't run leg 2.

## E-028 EVAL wave 5 · xl corridor AUC — DISPATCHED 2026-08-15 ~14:15Z

**RESULT (#333, landed 2026-08-15 23:53Z, ~8.7 h of chunked eval):
CONFIRMED — capacity keeps transferring to the roll at 205M. The xl tier
is the new roll champion by a wide margin.** Rolled corridor AUC: xl55 s0
**+0.664** / s1 **+0.663** / s2 **+0.664** (window +0.692/+0.690/+0.692;
gate scope 0.714/0.716/0.719). Gate e017_u1_s0 reproduced its pinned
+0.643 exactly — VALIDATION GATE PASSED — and read corridor +0.589, its
usual value, so the instrument is the same one that scored every previous
wave. Standings: xl55 0.6637 mean > big34 0.6237 > big55 0.6213 >
ring-8@222 champion 0.6043 > e017 0.589. That is **+0.042 over the big
tier** — the 576→768 capacity step bought +0.050 (#304), and 768→1024
bought +0.040 more: no saturation visible on the roll axis through 205M,
mirroring the forecast axis. The seed spread is astonishing: 0.001 across
three seeds (the forecast ratios spread 0.004), so at this capacity the
rolled skill is essentially deterministic given the data. The two E-026
stencil arms' inversion is now fully explained as capacity starvation:
with enough width, more input points help BOTH axes. Next questions this
number opens: (a) does the xl 60k→120k continuation move it further
(E-028b, waiting on the workflow refactor); (b) does xl89/xl144 at 200k
(#344–#347) beat 0.664 — landing today.

Question on record before the numbers: #304 showed capacity transfers to
the roll at 768×12 (+0.05 AUC paired vs 576×8, breaking the E-026
inversion). Does the trend continue at 205M, where the forecast ratios
(0.1295/0.1331) are the project's best? **Hypothesis**: xl corridor AUC >
the big tier's 0.621–0.628. **Falsifier**: xl ≤ big = the roll axis
saturates at ~88M even though the forecast axis doesn't, and the 120k
continuations (#326/#327) answer only forecast. Heads are the
weights-only e028xl55 publishes (opt state stripped for the 2 GiB cap —
evals need model+args only); gate e017_u1_s0 must reproduce 0.643.
Pinned behind #318 on gpu-box-45731106 (~17:30Z start, xl chunked eval is
the slowest yet — est. 8–12 h). Landed heads e029ar222_u1_s1/s2 and
e029bznoise_u1_s0 published this pass (znoise checkpoint's own args
confirm input_znoise=0.7); the E-029/E-030 eval wave dispatches when
their remaining arms land.

## E-028b · xl continuation 60k→120k — DISPATCHED 2026-08-15 ~12:10Z

**Why.** The xl arms (#308–#310, 205.4M) finished 60k NOT converged: val
z-MSE still falling at −0.007..−0.009 per 10k steps with lr at 3.7e-4,
and the final ratios (s1 0.13308, s2 0.12947 — new project bests, −0.016
vs big55) say the capacity curve hasn't bent at 205M. expdecay is
horizon-free (lr = peak·2^(−s/40000)), so an extension IS the
uninterrupted trajectory — commit 022f468 fixed the resume path, which
silently swapped expdecay for a fresh cosine on extension. #304's result
(capacity transfers to the roll, +0.05 AUC) makes the xl roll the most
promising open axis in the programme.

**Mechanics, for the record.** The mid-run head snapshots
(`run-<n>-temporal-latest.pt`) never reached the release for xl: a 205M
head with optimiser moments is ~2.5 GB, over GitHub's 2 GiB asset limit —
so the continuations resume the BOX-LOCAL mirrors
(`/opt/earth-cache/ckpt/run-{309,310}-temporal.pt`, written every metrics
point with opt/sched/RNG), each pinned to the box that trained that seed;
the `--resume-temporal <path>` rides the sched: tail because the
workflow's `resume2:` token cannot compose with stencil windows (its tag
parse takes everything after the colon — one-line fix deferred, needs the
PAT Workflows permission). Same-box pinning means they queue behind
#319/#321 (~2–4h) and start from files already on disk.

**Hypothesis**: another −0.005..−0.010 on the forecast ratio by 120k
(tail-slope extrapolation), and — the real question — a corridor AUC at
or above big-tier's 0.621–0.628 when the eval wave rolls the 120k heads.
**Falsifier**: 120k ratio within noise of 60k = converged, the extension
money buys nothing, don't extend again. Seeds 1–2 only (seed 0's box went
to E-029 work; two paired deltas answer the question).

**Also this pass**: #312 (E-029a s0) and #317 (E-029d s0) re-pinned off
their never-materialised Vast hosts (cancel + re-dispatch onto
gpu-box-47529389 / gpu-box-39184683 queues) — no new rentals; the two
hosts stay parked.

---

<a id="e-026"></a>
## E-027 · Scale analysis: transformer size × input width — DISPATCHED overnight 2026-08-14

**Why, from Chris.** *"Once you found the optimal arrangement (i'm rooting for
the sunflower, if confidence intervals overlap, please pick it), it's time to
do another scale analysis over night: 1) does a larger transformer help? 2) do
more input points (even further away, but sunflower style) help with the
larger transformer?"*

**The 2×2, three seeds per cell:**

| | sunflower-34 (35 slots) | sunflower-55 (56 slots) |
|---|---|---|
| **576×8 (standard)** | #282/#283/#284 (E-026) | **base55** #301/#302 (rerun, see incident 2) + #290 (s2, queued on fixed code) |
| **768×12 (2.7× params)** | **big34** #295/#296/#297 (rerun, see incident) | **big55** #298/#299/#300 (rerun) |

All four cells share the identical recipe otherwise: 60k steps, expdecay,
K=24, U=1, frozen run-62 codec, `spiral:111-4444-0.71-0.5`. The larger
transformer is d_model 768 × 12 layers vs 576 × 8 — ~2.7× stage-2
parameters, n_heads at the CLI default 4 either way (768 divides).

**"Even further away" was measured and REFUSED.** The occupancy sweep over
candidate 55-point sunflowers found that reach beyond 4444 km *deletes* live
input: 26.1 live corridor slots at r_max 4444, 22.7 at 5555, 19.7 at 6666 —
the extra reach lands off-window or on land and drags interior points outward
too. An 89-point sunflower at 5555 km holds 36.5 live slots but at 90 slots of
width (41 % occupancy). So more points: yes, 55 (next Fibonacci); further
away: no, the window is the binding constraint at 4444 km. If a future tensor
widens the window (global 0.25°), this decision should be remeasured.

**Hypotheses.** (1) The larger transformer does NOT help at 34 points — E-010
found stage-2 skill insensitive to capacity at these widths, and the codec is
frozen — but (2) it DOES unlock the 55-point width: E-022's width penalty was
sample efficiency, which capacity can buy back. Concretely: big55 > base55 and
big55 > big34, while big34 ≈ #282-284.

**Falsifiers.** big34 beating #282-284 by > 3 pooled seed sd falsifies (1) —
capacity was binding even at 34 points. base55 ≥ big55 falsifies (2) — width
is free or worthless regardless of capacity, and the interaction story dies.

**Shape choice is Chris's pre-commitment, recorded before the eval.** The
corridor-AUC eval of E-026 runs tonight in parallel; Chris pre-committed to
the sunflower if its CI overlaps the leader's. If the eval instead shows the
sunflower decisively BELOW the best arrangement, E-027 still answers its two
scale questions on a near-optimal shape, and the width/size winner gets
re-run on the winning arrangement — that cost is ~9 arms and is accepted in
exchange for the overnight not idling.

**Cost.** 3 base-sized arms ≈ 5 GPU-h + 6 large arms ≈ 15 GPU-h ⇒ ~20 GPU-h
≈ **$5.5**, queued one arm per box behind the draining E-026 queues.

**INCIDENT, ~22:00Z — the 768×12 arms died silently at the finish line, and
are re-dispatched fixed.** #285/#286 trained their full 60k steps cleanly
(~2 h each, the expected big-model pace) and then hit CUDA OOM the moment
eval 1 pushed its 20,000 held-out windows through the model in a single
forward — "tried to allocate 5.9 GB, 1.1 GB free", on two *different* boxes,
so config, not lemon. Both runs went green with no `temporal.json`: the
backgrounded-trainer silent-death signature, reproduced by a size change.
The one-shot eval batch was never a decision — it fit every 576×8 head ever
trained, so it never surfaced. `_chunked_forward` (4096-row slices,
numerically exact) now covers both 20k-window call sites; #287/#291–#293
were cancelled before wasting their own two hours; the six big arms rerun as
**#295–#297 (big34)** and **#298–#300 (big55)** on the fixed code. Cost of
the incident: ~4.5 GPU-h ≈ $1.2. The base55 arms (#288–#290, 576×8) were
never at risk and continue.

**INCIDENT 2, ~23:00Z — base55 was at risk after all, by a different route.**
"Never at risk" above was written about *activations* — 576×8 fits the 20k
one-shot on every head ever trained. What it missed is the *input* tensor:
20,000 windows × K=24 steps × **56 slots** × 64-d z is ~6.9 GB on device
before the model computes anything, and the 56-slot column is exactly what
this cell varies. #288/#289 trained their 60k steps cleanly and died at
eval 1 with the same green-run/no-temporal.json signature. `_chunked_forward`
(already merged for incident 1) covers this identically — the 4096-row slice
bounds input and activations together — so the fix needed no new code, only
re-dispatch: base55 s0/s1 rerun as **#301/#302**; #290 (s2) had not started
and picks up the fixed code at checkout. Incident cost: ~4 GPU-h ≈ $1.
Lesson appended to the OOM class: an eval batch scales with *stencil width*
as well as model size — any future input-shape change re-raises it, and
chunking is now unconditional so it cannot.

**FIRST RESULT, 00:17Z — the larger transformer helps, dramatically.**
#296 (big34 s1) landed green WITH temporal.json: d_model 768 / 12 layers
confirmed in the archive, wall 136 min (in family with #285/#286's healthy
60k-step pace — not a lemon), kfold 0.553, forecast ratio **0.14911**.
Paired at seed 1 the base cell (#283) is 0.17910 → **Δ = −0.0300**, ten times
the ~3-seed-sd falsifier bar (~0.009). One seed, but hypothesis (1) — "capacity
is not binding at 34 points" — is already dead unless #295/#297 reverse it,
and the best forecast ratio ever recorded anywhere in the project (previous:
0.17430) just fell by 0.025. E-010's "capacity doesn't matter" was measured
at K=6 with 1 slot; it does not survive K=24 × 35 slots. Await #295/#297
(same 2 h pace ⇒ ~00:20–00:45Z dispatch cohort lands through the night) and
the big55 cell before calling the interaction.

**RESULT (forecast axis), 01:40Z — all six arms landed green with
temporal.json; seeds and stencils verified from each archive's own
`stage2_result` (params_M disambiguates the stencil: 86.923M = 35 slots,
87.971M = 56 slots at 768×12; 34.098M = 56 slots at 576×8).** The 2×2 in
forecast ratio (lower = better), per seed and mean:

| cell | s0 | s1 | s2 | mean |
|---|---|---|---|---|
| base34 · 576×8 × 35 slots (#282–284) | 0.18093 | 0.17910 | 0.17692 | 0.17898 |
| base55 · 576×8 × 56 slots (#301/#302/#307) | 0.17860 | 0.17692 | 0.17380 | 0.17644 |
| big34 · 768×12 × 35 slots (#295–297) | 0.15186 | 0.14911 | 0.14827 | 0.14975 |
| **big55 · 768×12 × 56 slots (#298–300)** | 0.14997 | 0.14728 | **0.14562** | **0.14762** |

*(2×2 COMPLETED 09:20Z: #307 — the incident-3 re-run on fixed code —
delivered base55 s2 = 0.17380, seed/stencil verified in the archive
(34.098M, seed 2), head `e027base55_u1_s2` published. The width effect is
now confirmed paired at all three seeds at base scale too: −0.0023 /
−0.0022 / −0.0031, same sign every time, mean −0.0025 — slightly larger
than at 768×12 (−0.0021). Both main effects stand; still no interaction.)*

**Q1 — does a larger transformer help? YES, dramatically.** Paired at every
seed, big34 − base34 = −0.02907 / −0.02999 / −0.02865 (mean **−0.0292**, ~30×
the ~3-seed-sd bar). Hypothesis (1) is falsified three times over. The whole
E-026 shape table spans 0.010 from best arm to worst; capacity is worth three
times the entire geometry axis.

**Q2 — do more input points help with the larger transformer? YES,
consistently.** big55 − big34 paired: −0.00189 / −0.00183 / −0.00265 (mean
**−0.0021**, same sign at all three seeds). 55 sunflower points beat 34 under
768×12.

**But the INTERACTION story is dead: width is additive, not unlocked.**
base55 − base34 paired at the two finished seeds: −0.00233 / −0.00218 —
width helps by the SAME ~0.002 at 576×8 as at 768×12. Capacity did not "buy
back" a width penalty, because at this geometry there is no width penalty to
buy back: E-022's penalty was measured on TOUCHING 3×3 neighbours
(redundant, interpolable inputs), and a sunflower's 55 points at 111–4444 km
are not redundant. Two clean main effects — capacity ~−0.029, width ~−0.002
— and no detectable interaction. Hypothesis (2)'s prediction (big55 best)
was RIGHT, its mechanism (capacity unlocks width) was WRONG.

**Best model in the project as of 01:40Z: big55 s2 (#300), forecast ratio
0.14562** — 16% better than the pre-E-027 best (0.17412). Both scale axes
are OPEN at the top end: 768×12 is not a measured ceiling (E-028 candidate:
960×16), and neither is 55 points. Corridor AUC for all eight big/base55
heads pends the wave-4 eval; the E-026 shape decision is unaffected (it
compares base-scale arms under the pre-committed rule).

**Wall-times, all sane for their size** (fast=lemon check): big arms 136–200
min for 60k steps + chunked eval; base55 107–136 min; #281's 456 min is the
slow-box tax (same box that took 262 min for #284), not a config signal.

**EVAL wave 4 dispatched 02:04Z as #304** (box 47094145): rolled corridor
AUC for all eight landed scale heads — e027big34 s0–s2, e027big55 s0–s2,
e027base55 s0–s1 — against the e017 gate, heads published and
checkpoint-verified (`--expect-ring "spiral:111,4444,0.71,0.5"`,
`--expect-stencil 35/56`) at 01:50Z. Question on record before the numbers:
does the corridor AUC reproduce the forecast axis's two clean main effects
(capacity ~−0.029, width ~−0.002, additive)? E-022 is the standing warning
that a forecast gain can roll into an AUC loss; a big-arm AUC at or below
the base band would mean the capacity gain is one-step-ahead overfitting,
and the E-028 scale push dies before it is born. base55 s2 (#290, still
training) joins a later eval or is noted as the one silent cap.

**HARVESTED 2026-08-15 11:35Z — capacity transfers to the roll, and it
BREAKS THE INVERSION.** Gate: e017_u1_s0 reproduces 0.643 exactly, pass.
Rolled corridor AUC (flat mean msss_clim h1–12, 29,627 px):

| arm | s0 | s1 | s2 | mean |
|---|---|---|---|---|
| big34 (768×12, sun-34) | 0.6280 | 0.6180 | 0.6250 | **0.6237** |
| big55 (768×12, sun-55) | 0.6210 | 0.6210 | 0.6220 | **0.6213** |
| base55 (576×8, sun-55) | 0.5690 | 0.5730 | — | 0.5710 |
| e017 gate (576×8, stencil 1) | 0.5890 | | | 0.5890 |

Three findings. (1) **Capacity is worth +0.050 AUC paired by seed**
(big55−base55: +0.052 / +0.048) — the forecast gain is NOT one-step
overfitting; it rolls. The E-028 scale push lives. (2) **The E-026
anti-correlation was a CAPACITY-STARVATION artifact, not a property of
stencil geometry.** At 576×8 the sunflower arms rolled at 0.553–0.571,
WORSE than stencil-1's 0.589, while their one-step forecasts were the best
in the project — that was the inversion. Under 768×12 the same geometry
rolls at 0.621–0.628, BEATING everything: a small head fed 55×64-dim
inputs learns a one-step mapping whose errors compound under iteration; a
big head learns one whose errors don't. The channel/spatial structure
E-026b measured (slow subsurface pays, wind inverts) is the fingerprint of
that starved mapping, localized as the audit said in the
model-under-iteration. (3) **New AUC champions, +0.02 over ring-8@222's
0.6043**: big34 0.6237 > big55 0.6213 ≫ champion 0.6043 > e017 0.589.
big34 vs big55 is inside seed noise (ranges overlap); width beyond 34
points buys the roll nothing measurable, consistent with its tiny −0.002
forecast effect. Implication for the ON-HOLD E-026 pick: the pick was
scoped to base-scale arms and is now MOOT as a production decision —
production geometry should be re-decided at big scale, where #313/#314
(big-ring222) will say whether 8 points at 222 km still beat 34 spread to
4444 km once capacity is adequate. The eval wave for E-028 xl / E-029 /
E-030 heads is now the decisive experiment of the programme.

---

## E-026 · Ring of 8 vs ring of 16, in the TRANSFORMER — DISPATCHED 2026-08-14

**Why, from Chris.** *"I am not sure I trust the ridges. Linear is not enough.
Let's try our stage 2 transformer on the ring of 8 (already trained?) and
compare it with the ring of 16."*

Correct instinct, and the ridge has earned the suspicion. It has been wrong
about magnitude every time it has been checked against real training: it
predicted +0.63 % at one cell where E-022's 3×3 delivered ~0 %, and +1.6 % at
222 km where E-023's ring delivered ~3.9 %. It has only ever been reliable for
ORDERING, and the width question is precisely where its two predictions
disagree with each other:

- the **ridge** (400 centres) says 16 points ≈ 8 points, +0.0134 vs +0.0140 —
  a wider ring is redundant but harmless;
- **E-022** says the real model pays for width — 9 and 13 touching neighbours
  came out 6.3 and 8.1 seed sd WORSE than none, with
  `test_zero_weight_equivalence` proving it could have ignored them.

Both are predictions about the transformer and neither is the transformer.

**Hypothesis.** Doubling the ring to 16 points at the same 222 km radius does
not improve rolled corridor AUC, and may degrade it, because the extra eight
points are interpolations of the first eight (the ridge measures their joint
information as flat) while the input width doubles.

**Falsifier.** Three seeds of ring-16 whose corridor AUC exceeds ring-8's
0.6043 by more than 3 × the pooled seed sd (~0.005 → bar ≈ 0.619). That would
say width is cheap for this model after all and E-022's penalty came from
something specific to touching neighbours, not from width as such.

**Design.** `stencil:17,ring:222` — centre + 16 equidistant points on the same
222 km circle. Everything else is the e017/e023 recipe verbatim: 576/8 trunk,
60 k steps, U=1, K=24, expdecay. **Controls, both already trained and already
rolled by the same evaluator behind the same gate:** e023r222 (8 points, same
radius — the paired comparison, differing ONLY in point count) and e017 (no
neighbours). No new baseline runs are needed, which is what makes this cheap.

**Cost.** 3 arms × ~75–120 min ≈ 4.5 GPU-h, plus one 4-head evaluation
≈ 2.4 GPU-h ⇒ **~7 GPU-h, ~$1.9.**

**What it settles beyond itself.** If width is free in the real model, the
per-point framing still says to spend it well but removes the penalty term,
and E-024's whole line of reasoning weakens. If width costs, then the greedy
4-point stencil (more measured information at half the width) becomes the
clear next arm rather than one option among several.

### CORRECTION before any of it finished — "two rings, eight points each"

Chris, minutes after the first dispatch: *"to be clear, I meant two rings,
eight points each."* I had read "the ring of 16" as sixteen points on ONE
circle and dispatched `stencil:17,ring:222`, which is a different experiment:
**density at one radius**, not **reach across two**. #235 and #236 were
cancelled within minutes. **#234 is kept deliberately**, relabelled as a
single-seed DENSITY control — the forecast ratio reproduces to sd ≈ 0.002, so
one seed is readable on that metric, and having 8-at-222 vs 16-at-222 vs
8-at-222+8-at-555 separates "more points" from "more scales" for ~$0.40.

**The arms that answer the question as asked: #237 / #238 / #239**,
`stencil:17,ring:222-555` — centre + 8 points at 222 km + 8 at 555 km, the
outer ring **rotated half a sector** off the inner one so the shape samples
sixteen bearings instead of eight bearings twice. (Without that rotation the
far points sit directly behind the near ones, which is the one arrangement
guaranteed to buy less than it could.)

**This is the shape the ridge rated worst per point** (+0.0087 for sixteen
points against the single ring's +0.0123 for eight), so it is the sharpest
available test of Chris's objection to the ridge. Three outcomes, all
informative: the two-ring shape wins → the ridge's width penalty is an
artefact of the estimator and E-024's reasoning collapses; it ties → width is
free and the per-point framing is about efficiency, not skill; it loses →
the ridge and E-022 agree, and the narrow greedy stencil becomes the next arm.

**Status: DISPATCHED** — runs recorded below at dispatch time.

### EXTENDED — a factorial over scales and width (Chris: *"try a few different designs. Eg 3 rings, last ring at 1000"*)

Six more arms, chosen so the table separates the two things that have been
confounded all along — **how many scales** the input reaches across, and
**how wide** the input is:

| arm | runs | shape | slots |
|---|---|---|---|
| no neighbours | e017 (trained) | — | 1 |
| **one ring** | e023r222 (trained) | 8 @ 222 km | 9 |
| density control (n=1) | #234 | 16 @ 222 km | 17 |
| **two rings** | #237 / #238 / #239 | 8 @ 222 + 8 @ 555 | 17 |
| **three rings, wide** | #255 / #256 / #257 | 8 @ 222 + 8 @ 555 + 8 @ 1000 | 25 |
| **three rings, narrow** | #275 / #259 / #260 | 4 @ 222 + 4 @ 555 + 4 @ 1000 | 13 |
| **spiral of 13** | #261 / #262 / #263 | golden angle, 222 → 1000 km | 14 |
| **spiral of 8** | #264 / #265 / #266 | golden angle, 111 → 890 km | 9 |

The last two are the pair that matters most. They have **identical geometry —
same three scales, same maximum reach of 1000 km — and differ only in width**,
13 slots against 25. That is the per-point question asked in the transformer
instead of in the ridge:

- narrow ≈ wide → **width is what costs**, and every future design should
  spend its slots on distinct scales rather than on filling circles;
- wide > narrow → the extra bearings carry real information and the ridge's
  per-point framing (and my E-024 reasoning) is too pessimistic;
- both < the single ring → reach past ~222 km does not help this model at
  monthly cadence, whatever the width, and E-023's radius was the whole story.

Note the density control (#234, 16 points on ONE circle) sits at the same
slot count as the two-ring arm, so "more points" and "more scales" can be
read apart at fixed width.

**Cost of the extension:** 6 arms × ~1.5 GPU-h ≈ 9 GPU-h ≈ $2.5.

**RENUMBERED TWICE, 2026-08-14 — the live numbers are in the table below;
#240–#254 are dead.** Chris: *"Make sure to restart the second box or
decommission it, whatever is needed. More parallel boxes are fine, too."* The
stuck box was destroyed and four more rented, and the arms — all originally
pinned to `gpu-box-42005419`, which would have run the lot single-file for
~15 h while four boxes sat idle — were cancelled while still queued (they had
spent nothing) and re-dispatched one arm per box.

The second renumbering is worth recording, because it was **a real bug wearing
the costume of a known one**. The re-dispatch queued and stayed queued while
the API reported every new runner `online` and `idle` — which `ml/CLAUDE.md`
§2 tells you to read as a wedged runner and cure by cancel-and-re-dispatch.
It was not that. `runs-on:` matches **labels, never names**, and
`scripts/gpu_box.mjs` had never put the box's own name in `--labels`; the two
older boxes carry it only because someone added it by hand. So
`runner: gpu-box-46045353` matched nothing at all, and the documented cure
would have re-queued the arms into the same hole indefinitely, each cycle
looking like more evidence for the wrong diagnosis. Labels added to the live
runners, `gpu_box.mjs` fixed at registration, arms re-dispatched. Adding a
label does NOT rescue an already-queued job — measured; the match is decided
when the job is queued — which is why the numbers moved a second time.

**The fleet now runs five boxes, one arm each, three deep:** two rings on
`gpu-box-42005419`, wide on `-46045353`, narrow on `-47094145`, spiral-13 on
`-45731106`, spiral-8 on `-47566395`. ~4.5 h wall for fifteen arms instead of
~22 h, at ~$1.35/h across the fleet.

---

### SPIRAL — every point on its own bearing (Chris, 2026-08-14)

> *"One additional thing to try: angular coordinates should be different for
> each point, think of a spiral going outward. this may be the best design, as
> streams often flow straight, so it's important to catch 1-2 points for many
> incoming angles."*

This is a **physical** argument, and it names a defect every shape tried so
far shares. Rings reuse bearings: the three-ring wide arm spends 24 points on
16 directions, because rings 1 and 3 sit on the same eight. If what carries
the signal is *which direction the water is arriving from*, those duplicate
bearings bought nothing — they resolve a radius that is already resolved.

`--ring-km spiral:222-1000` puts point *k* at bearing *k* × **137.50776°**
(the golden angle, 360·(1−1/φ)) and at a radius growing geometrically from
r_min to r_max. The golden angle is not decoration: it is the unique rotation
for which **no prefix of the sequence ever clusters** — the phyllotaxis
arrangement — so the points are as evenly spread in bearing as that many
points can be, and stay so if the shape is later truncated.

**The point count is chosen, not rounded.** By the three-distance theorem a
golden-angle sequence leaves at most three distinct gaps, and the ratio of
largest to smallest collapses to **φ = 1.618 exactly at Fibonacci n** and is
**φ² = 2.618 for every other n** (measured over n = 4…29, exact to 1e-6;
pinned in `test_fibonacci_point_counts_are_the_uniform_ones`). A 12-point
spiral therefore has a blind sector 2.6× its own smallest gap — the one
defect the design exists to remove. So the arms carry **8 and 13** points,
even though 12 would have matched the three-rings-narrow slot count exactly.

**Arms.**

| arm | runs | shape | slots | bearings |
|---|---|---|---|---|
| **spiral of 13** | #261 / #262 / #263 | golden angle, 222 → 1000 km | 14 | 13 |
| **spiral of 8** | #264 / #265 / #266 | golden angle, 111 → 890 km | 9 | 8 |

The 13-point spiral is the matched twin of **three rings of four** (#275/#259/#260,
13 slots, same 222–1000 km reach): one extra slot, and 13 distinct bearings
against 8. The 8-point spiral is matched to the **champion** e023r222 (9
slots, 8 bearings, corridor AUC 0.6043) and differs from it in one thing only
— the ring puts all eight points at 222 km, the spiral spreads them 111 →
890 km. So the pair separates the two halves of Chris's idea: *more distinct
bearings* (spiral-13 vs narrow) and *more distinct radii at the same bearing
count* (spiral-8 vs the champion).

**Falsifier.** Three seeds of spiral-13 that fail to beat three-rings-narrow,
AND three seeds of spiral-8 that fail to beat the 0.6043 champion, would say
angular diversity is not what this model is short of, and send the next arm
back to radius rather than geometry.

**One infrastructure trap, caught before it cost anything.** `ml-train.yml`
parses the `ring:` field and then runs `RING="${RING//-/,}"` — dashes are how
a multi-radius list survives an input whose own fields are comma-separated —
so `ring:spiral:222-1000` arrives at `temporal.py` as `spiral:222,1000`, and
a `split("-")` parser would have raised **after the embedding**, six GPU-hours
in. Fixed on the parser side (both separators accepted) rather than in the
workflow, because `ml-train.yml` sits exactly at the 25-input
`workflow_dispatch` ceiling and a 26th makes every dispatch in the repo 422.

---

### DEEP SPIRALS — 24 and 36 points reaching 4444 km (Chris, 2026-08-14) · **PRIORITY**

> *"Please add and prioritize a 24 and a 36 point spiral experiment as well
> (each point at a different angle, and farthest point up to 4444km away)."*

**Why 4444 km is a bigger change than the point count.** E-022's standing
physical caveat is that one roll step is ONE MONTH, so a stencil that reaches
1–2 cells reaches 1–2 cells *per month* — while the Gulf Stream advects
**100–200 cells per month**, which at 0.25° is **2800–5600 km**. Every shape
tried so far, up to and including the 1000 km three-ring arm, is therefore
*structurally* unable to see where this month's water came from; the coupling
they buy is slow interior dynamics, not advection. **A 4444 km reach is the
first geometry in this programme that can hold a month of Gulf Stream inside
itself.** That is a different hypothesis from "more bearings help", and these
two arms test it.

| arm | runs | shape | slots | points | bearings (≥10° / distinct) |
|---|---|---|---|---|---|
| **spiral of 24** | #267 / #268 / #269 | golden angle, 111 → 4444 km | 25 | 24 | 21 / 24 |
| **spiral of 36** | #270 / #271 / #272 | golden angle, 111 → 4444 km | 37 | 36 | 21 / 36 |

Both start at 111 km and end at 4444, so **24 vs 36 is purely density at fixed
reach**, and each is comparable to the shallower spirals through its shared
r_min. The 24-point arm also sits at **exactly the slot count of the
three-rings-wide arm** (#255–#257, 25 slots, 1000 km, 16 bearings) — same
width, 4.4× the reach, 24 bearings instead of 16.

**Hypothesis.** Reach, not width and not bearing count, is the binding
constraint: a stencil that spans a month of advection beats the 222 km
champion's 0.6043 corridor AUC by more than the pooled seed sd, and the
36-point arm adds little over the 24-point one because the extra twelve points
resolve angle the model was not short of.

**Falsifier.** Three seeds of spiral-24 at or below 0.6043. That would say the
monthly-cadence model cannot use distant information *at all* — which, given
that E-023 found information peaking at 222 km and decaying outward, is the
live alternative and would close the whole "reach" line rather than just this
arm.

**Precondition, checked before any GPU was spent** (`ml/CLAUDE.md` §0.3).
4444 km is ~40° of latitude or ~52° of longitude and the family3 window is
only 70°×120°, so the far slots could have been mostly off-window or on land —
where `build_stencil` writes −1 and `gather_stencil` substitutes exact zeros.
That design would be a *narrow* shape paying a wide shape's parameter count,
and it would have trained perfectly happily while being that.
`ml/measure_slot_occupancy.py` measures it against the evaluator's own mask:

| design | slots live, all rolled px | slots live, corridor px |
|---|---|---|
| ring of 8 @ 222 km (champion) | 88.2 % (7.1/8) | 88.2 % (7.1/8) |
| three rings 8+8+8 @ 1000 km | 79.0 % (19.0/24) | 77.0 % (18.5/24) |
| **spiral of 24 @ 4444 km** | **70.4 % (16.9/24)** | **68.1 % (16.4/24)** |
| **spiral of 36 @ 4444 km** | **70.6 % (25.4/36)** | **68.6 % (24.7/36)** |

Thinner than the champion, but real and usable — the check clears, and the
number is on the record so a weak result can be read against it rather than
explained by it afterwards.

**One deliberate deviation from the shape's own optimum, recorded not
silently fixed.** 24 and 36 are not Fibonacci numbers, so their bearing gaps
have ratio φ² = 2.618 rather than φ (21 and 34 would be the φ-uniform counts).
At 36 points that puts some bearings 4.8° apart. It is recorded rather than
substituted because at *these* radii a 4.8° gap is still 335 km of separation
at the outer edge — the near-clustering that makes 12 a bad count for a
1000 km spiral is not the same defect at 4444 km.

**Cost.** 6 arms × ~1.5 GPU-h ≈ 9 GPU-h ≈ $2.6, on two boxes rented for them
so they run *beside* the factorial rather than behind it.

---

### EARLY READS — full standings at 22:35Z, forecast ratio (corridor AUC pends the evals)

Every E-026 arm complete except spiral-34 s2 (#281, in flight). Ranked by
mean; per-seed columns are directly comparable DOWN the table (the seed-index
effect makes rows comparable only within a column):

| rank | arm | reach km | pts | s0 | s1 | s2 | mean | vs champion |
|---|---|---|---|---|---|---|---|---|
| 1 | **spiral-34, geometric** | 4444 | 34 | 0.17702 | 0.17501 | **0.17209** | **0.17471** | **−0.0101** |
| 2 | **elliptic-24 ×0.71** | 4444 | 24 | 0.17853 | 0.17563 | 0.17471 | 0.17629 | −0.0085 |
| 3 | three rings wide 8+8+8 | 1000 | 24 | 0.17928 | 0.17592 | 0.17412 | 0.17644 | −0.0083 |
| 4 | spiral-24, geometric | 4444 | 24 | 0.17943 | 0.17742 | 0.17430 | 0.17705 | −0.0077 |
| 5 | **sunflower-34 (far-heavy)** | 4444 | 34 | 0.18093 | 0.17910 | 0.17692 | 0.17898 | −0.0058 |
| 6 | two rings 8+8 | 555 | 16 | 0.18353 | 0.17931 | 0.17784 | 0.18023 | −0.0045 |
| 7 | spiral-13 | 1000 | 13 | 0.18260 | 0.18140 | 0.17686 | 0.18029 | −0.0045 |
| 8 | three rings narrow 4+4+4 | 1000 | 12 | 0.18296 | 0.18162 | 0.17817 | 0.18092 | −0.0038 |
| 9 | spiral-8 | 890 | 8 | 0.18465 | 0.18216 | 0.17906 | 0.18196 | −0.0028 |
| 10 | ring 16 @ 222 (density, n=1) | 222 | 16 | 0.18545 | — | — | 0.18545 | +0.0007 |
| — | champion 8 @ 222 (e023r222) | 222 | 8 | | | | 0.18476 | 0 |
| — | no neighbours (e017) | 0 | 0 | | | | 0.19216 | +0.0074 |

*(Table completed 01:40Z: #281 landed spiral-34 s2 = 0.17209 — the best
576×8 forecast number ever recorded, beating the previous best single ratio
0.17412 by 0.002 and pulling spiral-34's mean to 0.17471. The paired order
below is unchanged; spiral-34 is now best at all three seeds.)*

The paired-by-seed ordering inside the 4444 km family is CONSISTENT at every
seed: **spiral-34 ≤ elliptic-24 ≤ spiral-24 < sunflower** — so on one-step
forecast the ramp verdict reads "keep the near field", the aspect is worth a
small positive nudge (elliptic beats circular spiral-24 at s0 and s1, ties at
s2), and 34 points beat 24 at the same reach. The sunflower's s2 (0.17692)
closed much of its gap; its penalty is real but shrinking with seed,
consistent with the dead slots costing sample efficiency. Reach remains the
dominant axis: every 4444 arm beats every ≤1000 arm except wide, which holds
rank 3 — 1000 km with a dense outer ring remains remarkable value per slot.

**Decision rule AMENDED by Chris, 2026-08-14 ~23:30Z** — *"I am happy with
your pick of stencil shape. Let's use whatever comes out best on AUC, or
rather, you can judge and decide yourself while I am sleeping."* The
sunflower tiebreak is withdrawn; the pick is the best rolled corridor AUC by
paired-by-seed comparison, with judgment applied if the top arms are
statistically indistinguishable — in that case secondary evidence (forecast
metric, live-slot efficiency, the CFL/physics argument) decides, and the
reasoning is written into the RESULT rather than exercised silently.

Heads on the release: tworing, wide, narrow, sp13, sp8, sp24, esp24, sun34
(×3 each), sp34 s0+s1, ring16. Eval wave 1 (#294: gate +
sp24 + wide + champion) lands ~02:00Z; wave 2 (sunflower + elliptic +
spiral-34) follows on the second eval box. Earlier per-seed detail and the
seed-mislabel correction are preserved in git history (commits 324c796,
e3bcb0e).

**Waves 3a/3b dispatched 02:04–02:08Z as #306/#305** (boxes 45731106
(revived from self-exit) and 47529389), splitting the remaining arms across
two freed boxes so the WHOLE table lands by morning instead of noon: 3a =
gate + tworing ×3 + narrow ×3; 3b = gate + sp13 ×3 + sp8 ×3 + ring16 (n=1) +
**sp34 s2** — #281 landed at 01:00Z with forecast ratio **0.17209**, the
best 576×8 number ever recorded, completing the forecast leader's third
seed (head `e026sp34_u1_s2` published 01:47Z, checkpoint-verified seed 2 /
stencil 35 / `spiral:111,4444`). With waves 1+2 this is every E-026 arm,
every seed, no silent caps: 33 head-evals across five boxes.

### CORRIDOR AUC, wave 1 (#294, landed 02:13Z) — the reach arms roll WORSE than no neighbours

Gate PASSED: e017_u1_s0 reproduces horizon_auc 0.643 exactly (tol 0.0101),
and #294's corridor read for the champion s0 (0.608) matches #233's to the
third decimal — the evaluator is stable across boxes and days. Metric =
`horizon_auc` on the corridor subset (mean MSSS-vs-climatology over h=1–12;
`auc_damped` in parentheses as the secondary):

| head | s0 | s1 | s2 | mean |
|---|---|---|---|---|
| champion 8@222 (e023r222, #233) | 0.608 (0.596) | 0.599 (0.587) | 0.606 (0.594) | **0.6043** |
| e017 no-neighbours s0 | 0.589 (0.577) | | | 0.589 |
| three rings wide 8+8+8 @1000 | 0.571 (0.559) | 0.584 (0.573) | 0.589 (0.578) | 0.5813 |
| spiral-24 @4444 | 0.572 (0.561) | 0.584 (0.573) | 0.578 (0.567) | 0.578 |

**E-022's warning fires again, at full scale.** Spiral-24 was 0.0077 better
than the champion on one-step forecast; rolled twelve months over the
corridor it is 0.026 WORSE, and even sits below the no-neighbour baseline at
every seed but s1-vs-e017-s0. Same for wide. The one-step forecast axis and
the rolled-AUC axis are ANTI-correlated across these arms — reach helps the
model predict next month and hurts what compounds. A plausible mechanism:
far-field inputs let the head fit advective detail that is right at h=1 and
wrong by h=6, where a 222 km ring can only smooth. The decision rule reads
corridor AUC, so as of wave 1 the standing champion (ring-8 @ 222 km) is
still the best stencil measured. Waves 2/3 (sunflower, elliptic, spiral-34,
and the 1000-km-and-under family) decide whether ANY E-026 arm beats it —
the near-field arms (tworing, sp13, sp8, ring16) are now the live
candidates, exactly the opposite of what the forecast table suggested.

### CORRIDOR AUC, wave 2 (#303, landed 05:39Z) — the deciding arms confirm it: rolled skill falls monotonically with reach

Gate PASSED again (e017 → 0.643 exactly, third eval in a row). The wave-2
arms — the three forecast leaders Chris and I spent the day designing — all
roll WORSE than wave 1's:

| head | s0 | s1 | s2 | mean |
|---|---|---|---|---|
| elliptic-24 ×0.71 @4444 | 0.569 (0.557) | 0.563 (0.551) | 0.572 (0.560) | 0.568 |
| spiral-34 @4444 | 0.551 (0.540) | 0.573 (0.562) | *(wave 3b)* | 0.562² |
| sunflower-34 @4444 | 0.557 (0.545) | 0.559 (0.548) | 0.543 (0.531) | 0.553 |

²two seeds until #305 lands.

**The full ordering as of wave 2, by corridor AUC mean:** champion 8@222
**0.6043** > e017 no-neighbours 0.589 > wide@1000 0.5813 > sp24@4444 0.578 >
esp24@4444 0.568 > sp34@4444 0.562² > sun34@4444 0.553. Rolled corridor
skill is **monotone decreasing in reach** (222 → 0 → 1000 → 4444) and,
within 4444 km, decreasing in point count and far-weighting — the exact
inverse of the forecast-ratio table, where sp34 leads and the champion
trails by 0.010. The two objectives are not merely uncorrelated across
stencil geometry; they are anti-correlated, and the effect (−0.05 AUC from
best to worst) is ten times the seed noise (~0.005). The forecast axis
optimises next-month fidelity; the roll rewards whatever stays stable when
its own output is its input. A wide stencil gives the head detail it trusts
at h=1 and compounds at h=6; e017's stencil-of-one cannot even see the
neighbouring pixels' drift, and the 222 km ring adds just enough context to
correct locally without importing far-field noise. Waves 3a/3b (tworing 555,
narrow/sp13/sp8 ≤1000, ring16 222) fill in the near-field candidates — on
this curve the interesting question is whether ring16@222 (denser at the
champion's own radius) can move 0.6043.

### CORRIDOR AUC, wave 3a (#306, landed 06:44Z) — first arms to MATCH the baseline; none beat the champion

Gate PASSED (fourth eval, e017 → 0.643 again). The near-field factorial
arms are the first E-026 geometry to climb back to the no-neighbour line:

| head | s0 | s1 | s2 | mean |
|---|---|---|---|---|
| three rings narrow 4+4+4 @222–1000 | 0.583 (0.571) | 0.591 (0.580) | 0.593 (0.582) | **0.589** |
| two rings 8+8 @222–555 | 0.590 (0.578) | 0.588 (0.577) | 0.579 (0.568) | 0.5857 |

narrow exactly MATCHES e017's 0.589 on the mean and beats it at two of
three seeds — the first stencil since the champion to not pay for its
neighbours on the roll. But paired against the champion (0.608/0.599/0.606)
both arms lose at every seed: tworing by −0.018/−0.011/−0.027, narrow by
−0.025/−0.008/−0.013. The reach curve now reads: 222 km ring (0.6043) >
{narrow ≤1000 sparse, e017 nothing} (0.589) > tworing ≤555 (0.586) >
wide@1000 dense (0.581) > everything @4444 (0.578→0.553). Remaining
candidates in #305: sp13/sp8 (≤1000 spirals), ring16@222 (the density arm
at the champion's own radius, n=1) and sp34 s2 (completes the forecast
leader's AUC row).

### CORRIDOR AUC, wave 3b (#305, landed 07:29Z) — the base-scale table is COMPLETE; two clean isolations for the audit

Gate PASSED (fifth eval, e017 → 0.643). New rows: sp13 0.585/0.595/0.579
(mean 0.5863) · sp8 0.591/0.589/0.585 (**0.5883**, the best new arm —
statistically at the baseline) · ring16@222 s0 **0.564** (n=1) · sp34 s2
0.574 (completing sp34 at 0.566). The full E-026 table by corridor-AUC
mean, all gates passed, all seeds paired-comparable:

| rank | stencil | pts | reach km | AUC mean | vs champion |
|---|---|---|---|---|---|
| — | **champion ring-8 @222 (E-023)** | 8 | 222 | **0.6043** | 0 |
| 1 | spiral-8 | 8 | 890 | 0.5883 | −0.016 |
| 1 | e017 no neighbours | 0 | 0 | 0.589 | −0.015 |
| 1 | three rings narrow | 12 | 1000 | 0.589 | −0.015 |
| 4 | spiral-13 | 13 | 1000 | 0.5863 | −0.018 |
| 5 | two rings 8+8 | 16 | 555 | 0.5857 | −0.019 |
| 6 | three rings wide | 24 | 1000 | 0.5813 | −0.023 |
| 7 | spiral-24 | 24 | 4444 | 0.578 | −0.026 |
| 8 | elliptic-24 | 24 | 4444 | 0.568 | −0.036 |
| 9 | spiral-34 | 34 | 4444 | 0.566 | −0.038 |
| 10 | ring-16 @222 (n=1) | 16 | 222 | 0.564 | −0.040 |
| 11 | sunflower-34 | 34 | 4444 | 0.553 | −0.051 |

The champion beats every arm paired at every seed. **Two isolations the
audit can use, measured not hypothesized:** (1) point count at FIXED
radius — ring16@222 vs ring8@222, same 222 km, doubled points, −0.044 at
s0; (2) reach at FIXED count — sp8@890 vs ring8@222, same 8 points, wider
reach, −0.016. Within the spiral family the AUC is monotone in count
(8→13→24→34 = 0.588→0.586→0.578→0.566). Whatever the roll penalizes, it
scales with BOTH slot count and reach, and count at least as strongly —
the ring16 datum kills any story that is purely about physical distance.
Per Chris's directive the PICK remains ON HOLD pending E-026b (a)+(b) and
the #304 capacity AUCs; no RESULT is written yet.

### E-026b · AUDIT of the anti-correlation (Chris, 08-15 morning: "investigate thoroughly and not hypothesize — there could still be an issue in how we compute AUC")

**The pick is ON HOLD pending this audit and #304.** The two mechanism
paragraphs written overnight (waves 1–2 above) are hereby demoted to
conjecture; below is what has been MEASURED, all from the four archived
rollout JSONs (#233/#294/#303/#306, 27 head-evals) — zero GPU spent.

**Artifact classes eliminated:**

1. **Different scored sets per stencil — NO.** Every one of the 27 heads has
   IDENTICAL sample counts at every horizon (h1 n = 37,528,668 …
   h12 n = 3,127,389, corridor). The evaluator scores the same pixels,
   months and channels for a 1-slot head and a 56-slot head.
2. **Broken input construction in the roll (dead slots, boundary handling)
   — NO.** At h=1 the rolled context IS observed truth, so the roll's input
   path is exercised with training-equivalent inputs — and the h=1 corridor
   msss reproduces the training forecast ordering exactly: sp34 0.737 >
   sp24 0.736 > wide 0.735 > … > champion 0.727 > e017 0.713. If slot
   handling differed from training, the far-reach heads would already
   suffer at h=1. They lead at h=1.
3. **Baseline choice — NO.** The ordering is identical against climatology
   and against persistence, and under the damped variant.
4. **Amplitude collapse masquerading as skill loss — NO.** amp_ratio does
   not track the inversion: sp34 holds amplitude best-in-class at h=1
   (0.888) AND near-best at h=12 (0.766) while showing the fastest msss
   decay. The decay is in ACC (pattern correlation): sp34 0.858→0.683
   vs champion 0.852→0.730. The far-reach heads predict confident,
   full-amplitude, increasingly WRONG patterns.
5. **Horizon weighting of the AUC — TOP IS INVARIANT.** Flat mean vs
   n-weighted mean (which favours early horizons and thus the far-reach
   heads): the champion's three seeds are ranks 1–3 under both. The
   mid-table (arms within ~0.01 of the baseline) DOES shuffle under
   re-weighting — differences at that scale should not be read.

**What the data says without hypothesis:** the inversion is not present at
entry (h=1 matches training) and develops smoothly with horizon; crossover
at h≈3–4; by h7–12 the spread is 0.564 (champion) vs 0.478 (sp34 s0). It
is a pattern-error phenomenon under iteration, not an accounting artifact
in the metric, in the masks, or in the input plumbing.

**Remaining checks (next, in order of cost):** (a) per-CHANNEL horizon
curves — the aggregate could hide a channel subset driving the divergence
(late-starting channels, wind stress); needs a lightly instrumented eval
re-run on 3 heads (champion / sp34 / e017). (b) SPATIAL skill maps at h=6 —
where in the corridor the far-reach decay lives (boundary-adjacent pixels
whose stencils reach off-window vs interior); same instrumented run.
(c) the `long` (240-month) and `future` blocks of the existing archives —
consistency of the same ordering on an independent protocol. (d) exact
month-set assertion per head (implied by n-equality; cheap to assert
exactly). No further conclusions until (a)+(b) are measured.

**PHASE 2 (#311, landed 10:10Z) — (a) and (b) measured.** The instrumented
eval re-rolled e017 / champion / sp34-s1 with per-channel corridor curves
and the h=6 per-pixel window map. Trust checks first: gate PASSED (sixth
eval), the recomposition identity holds on all three heads
(identity_max_dev ≤ 0.0005 vs the 0.002 bar), and all three corridor AUCs
reproduce their earlier evals to the third decimal.

**(a) The divergence is CHANNEL-STRUCTURED, not broad-spectrum.** At h=6,
sp34 loses to the champion by ~0.10 msss on the slow subsurface fields —
rg_t400/500/700 and rg_s300–500, the 300–700 m heat/salt reservoir — with
29 of 39 channels negative. But SIX channels run the OTHER way: sp34 BEATS
the champion at h=6 on the wind-stress channels (tau_x +0.099, tau_y_std
+0.075, tau_y +0.043) and rg_t100 (+0.059). The far-reach stencil keeps
its advantage on fast atmospheric forcing under iteration and pays on the
slow ocean-memory fields — which are precisely the AMOC-relevant ones, and
precisely the fields where climatology is hardest to beat at depth.

**(b) The decay does NOT live at the boundary — it lives where the far
stencil is FULLY LIVE.** Per-pixel dead-slot fraction of the sp34 stencil
vs its h=6 penalty: corr = **+0.176**, monotone in quintiles — pixels
whose stencils are 0–12% dead (mid-Atlantic interior) carry the largest
penalty (mean Δ −0.061), pixels whose stencils are ≥47% dead roll at
PARITY (+0.003). The boundary/dead-slot artifact hypothesis predicted the
opposite sign; it is dead (sixth artifact class eliminated). The map
(figure with #311) shows the effect as a mottled band through the open
Atlantic interior, near-white along coasts and marginal seas.

**What the audit now supports saying, as measurement:** the rolled-skill
penalty of wide/far stencils (i) is absent at h=1, (ii) grows with
horizon, (iii) scales with slot count and reach, (iv) scales with the
amount of LIVE far-field input actually consumed, (v) lands on the slow
subsurface channels while sparing — even favouring — the fast wind
channels, and (vi) is robust to every metric-accounting alternative
tested (masks, baselines, amplitude, weighting, identity). This localises
the phenomenon in the model-under-iteration, not the evaluator. The
E-029 arms are the discrimination experiments for the remaining candidate
mechanisms: if input-noise training (b) or U=2 (c) closes the gap, the
train/roll input mismatch was the cause; if neither moves it, the
explanation must live elsewhere (e.g. in what the objective optimises at
depth), and that is a finding too.

### E-029 · Reuniting the two axes — DISPATCHED 08-15 ~09:45Z

Chris approved the full proposal slate ("All sounds good. Also curious
about U=2, as well as all your proposals. Consider adding more points
(sunflower style), too."). Four arms families, all at the 768×12
transformer that E-027 showed is worth 3× the whole geometry axis:

**(a) big-ring222 ×3 — the production candidate.** 768×12 on the champion
ring-8@222 (stencil 9), seeds 0–2. The AUC champion has only ever been
trained at 576×8; the capacity gain has only ever been measured on
sunflowers. Hypothesis: the capacity effect transfers (forecast ratio well
below e023r222's 0.18476, plausibly ~0.155 if the −0.029 is
geometry-independent). Falsifier: ratio ≳ 0.18 = capacity needs width to
pay, and the interaction that was dead on the width axis lives on the
reach axis. The rolled corridor AUC — the number that decides whether this
IS the production model — comes from a later eval wave.

**Harvested (#313 s1, #314 s2, 2026-08-15): ratio 0.15278 / 0.15088.**
Capacity transfers to the champion geometry too: −0.033 vs base ring222's
0.18476, right on the ≈−0.030 prediction. Note big-ring222 (~0.152) reads
WORSE on forecast than big55 (0.1476) — same direction and size as the
width effect at base scale — so the forecast axis now says wide-and-far
beats narrow-and-near at BIG scale as well; the rolled AUC (eval wave,
with #324's s0) decides whether ring222's roll advantage survives adequate
capacity, which #304 predicts it will not.

**(b) znoise ×2 — attack the train/roll gap directly** (code pending, dispatched
after it lands): Gaussian noise on the input z during training, σ set from
the model's own measured one-step error (√val_zmse ≈ 0.7 z-units), so
training-time context statistically resembles roll-time context.
Hypothesis: forecast ratio worsens slightly, rolled AUC improves.
**Harvested (#320 s0, 2026-08-15): ratio 0.15480** vs clean big55's
0.14762 — worse by 0.007, as predicted (noise makes the one-step task
harder). The AUC eval decides whether the robustness pays where it
matters. rapid_r_kfold 0.553 [0.429, 0.651], the best single-head kfold
seen on a big-tier arm.

**(c) U=2 × wide ×2 — the interaction E-010 could not test.** E-010's
"unroll buys nothing" (settled negative, §8) was measured at STENCIL 1,
before stencils existed; unroll's mechanism is feedback through the
inputs, which a 1-slot model barely has. U=2 on big55, seeds 0–1, paired
against #298/#299. The forecast ratio will read WORSE (E-010's 29.7%
at U=4 says so); the question is the later AUC. Falsifier for "unroll
fixes wide stencils": corridor AUC ≤ big55's own.

**(d) sunflower-89 ×3 — the width axis continues.** Next Fibonacci; r_max
stays 4444 (E-027 occupancy refusal). Pre-registered occupancy: **47.1%
corridor (41.9/89 live)** — the same live fraction as sunflower-34 (47.6%)
and 55 (46.6%), so more points = proportionally more live input.
Hypothesis: another paired ~−0.002 vs big55 (width effect was −0.0025/
−0.0021 at 34→55); falsifier: within seed noise (~0.0015) = width
saturates at 55.

Windows note for the record: `unroll:` is a PREFIX match in the workflow
while `stencil:/ring:/seed:` match anywhere, so the U=2 windows begin
`unroll:2,stencil:56,…`. Cost: 10 arms ≈ 32 GPU-h ≈ $9.5, plus the later
AUC eval wave that arms (b) and (c) exist for.

**CORRECTION, 10:40Z — arms (c) RETRACTED: U>1 × stencil>1 is
architecturally impossible as dispatched, and I dispatched it anyway.**
temporal.py guards exactly this (line ~1210): the training-time unroll
feeds back only the CENTRE pixel's predicted z, and at u≥1 the neighbour
slots would need the NEIGHBOURS' own predictions, which a random-pixel
batch does not contain — the shapes do not even align (pred is [B,K,d_z],
the wide input needs [B,K,S·d_z]). E-010 never met the guard because it
ran at stencil 1. #315 refused at startup and went GREEN anyway (the
backgrounded-trainer signature, ~40 min of box setup wasted); #316
cancelled before starting. This is a §1 violation on my part — "check the
configuration can produce it" — caught by Chris asking how missing slots
are encoded, which put my eyes on the neighbouring code block. A TRUE
wide-stencil unroll requires predicting the neighbours too, i.e.
field-level training (rollout-with-gradients over tiles or the window) —
recorded below as the E-030 candidate; feasible cheaply only for
short-reach stencils (ring-8@222 = ±2 cells → tile batches). Until then
the exposure-bias discrimination rests entirely on arms (b), which are
architecturally fine precisely because znoise SIMULATES predicted-input
degradation statistically instead of needing actual neighbour predictions.

**Missing-slot encoding, documented while answering Chris's question
("does a missing pixel need a special symbolic token?").** A dead slot
(off-window or land) is zero-filled in z-space by `gather_stencil` — no
in-band token — but the model is TOLD the deadness explicitly and
out-of-band: `static_ctx` = [Zstat · coords · obs_flags], where obs_flags
is one binary per slot (1=live, 0=dead) and coords is the pixel's own
position. So "the model knows west/east/middle" is true BY DESIGN, through
flags and coordinates, not inferred from the zeros; the zero-vs-real-value
collision is disambiguated by the flags; and the encoding is identical at
train and roll time (geometry is static), so it opens no train/roll gap.
The audit's phase-2 finding cross-checks it from the other side: the
inversion is WORST where nothing is missing, so missing-slot handling is
measurably not the driver. A learned missing-embedding (the true "token"
analogue — slots are concatenated features per timestep, not transformer
tokens) remains a cheap testable variant if the flags ever prove
insufficient.

### E-028 · Even bigger transformers — DISPATCHED 08-15 ~07:15Z

Chris: *"let's try even bigger transformers."* **xl55 = 1024×16 (~207M
stage-2 params, 2.4× big55's compute)** on the same sunflower-55 stencil so
the capacity axis stays controlled: 576×8 → 0.17776 (2 seeds) → 768×12 →
0.14762 → 1024×16 → ? Three seeds: **#308/#309/#310** (boxes 45731106,
47094143, 47566395 — two revived from self-exit). Hypothesis: the forecast
ratio improves again — the 768×12 step showed no saturation. Falsifier:
mean within ~3 seed sd (0.005) of big55's 0.14762 = the curve is
flattening. The corridor-AUC standing of ALL capacity cells pends #304 and
E-026b; which stencil the big models should live on is decided after both.
~5–6 h/arm ≈ 16 GPU-h ≈ $4.5.

### INCIDENT 3 (#290): a queued run executes the sha of main AT DISPATCH, not at start

#290 (base55 s2) went green with **no temporal.json**: 60k clean training
steps, final in-training probe written (rapid 0.529), then death at the
one-shot 20k-window eval — the same 56-slot input-tensor OOM as #288/#289.
It was dispatched at 18:59Z, six hours before it started running: the
`_chunked_forward` fix was on main by 22:30Z and #290 began at ~01:00Z, but
a workflow run pins the commit **at dispatch time** (provenance sha 69ea03a,
18:22Z — pre-fix). Pass 6's note "picks up the fixed code at checkout" was
wrong, and is exactly the kind of assumption §0.1 exists for: the artefact
(provenance.json's sha) says what ran; the intention does not. **Lore: a fix
merged while a run sits queued does NOT reach that run — cancel and
re-dispatch anything queued at a bad sha.** Re-dispatched on fixed code as
**#307** (box 30257785, warm from the wave-2 eval). Incident cost: ~3.5
GPU-h ≈ $1 and the base55 cell waiting on its third seed until ~08:30Z.

### HOW A STENCIL ROLLS FORWARD — the design theory behind the deep and elliptic arms (2026-08-14)

Chris: *"think through the rolling forward predictions with the model. How
should points be arranged best to roll forward over multiple months?"*

**The governing constraint is the CFL condition, imported from numerical
PDEs.** The evaluator advances the whole window one month per step, and each
pixel's next state is computed from its stencil. Information therefore
propagates through the rolled field at **at most one stencil-reach per
month** — the numerical domain of dependence. For the roll to even in
principle track moving water, that speed must cover the flow's own:

| dynamics | speed | km/month | months for a 222 km stencil to keep up | for 4444 km |
|---|---|---|---|---|
| Gulf Stream core | 1–2 m/s | 2600–5200 | never (12–23× too slow) | ~1 step |
| North Atlantic Current | 0.2–0.5 m/s | 520–1300 | 2.3–5.9× too slow | ~1 step |
| interior / recirculation | 2–5 cm/s | 52–130 | keeps up | keeps up |
| Rossby waves (westward) | ~2–5 cm/s | 52–130 | keeps up | keeps up |

This reframes the whole shape programme: E-023's 222 km champion wins by
coupling the *slow* rows of that table, and is structurally blind to the fast
ones. The 4444 km spirals are the first shapes whose numerical domain of
dependence contains the jet — which is why the reach arms are the priority,
before density, before bearing count.

**Two routes for multi-month information, and the ramp serves both.** A
horizon-h prediction can get its far-field information (a) *compositionally* —
h roll steps × reach, but every step after the first reads *predicted* states,
so error compounds along the path — or (b) *directly* — the first step reads
the observed state at distance ≈ flow × h, no compounding, but the model must
internalise the whole transport path in one map. The geometric radius ramp is
what lets one shape serve route (b) for every horizon at once: log-spaced
radii are log-spaced horizons at fixed flow speed, and at fixed radius they
are log-spaced horizons across the speed spectrum — 4444 km is ~1 month of
jet, ~4 months of NAC, and ~3–7 *years* of interior drift. The same ramp that
E-024 justified by information-per-point is also the horizon ladder.

**Why an ellipse, and why 0.71 exactly.** Both zonal directions carry future
information — mean advection in the jet arrives from the west, Rossby-wave
anomalies propagate in from the east — while coherent meridional transport at
these scales is weaker. So the window should be wide rather than tall, but
*how much* wider is an empirical question, and `ml/measure_flow_anisotropy.py`
answers it from the tensor's own SSH channel (geostrophic u,v; the per-channel
global z-score makes the ratio exact):

| population | flow | mean\|u\|/mean\|v\| | rms ratio |
|---|---|---|---|
| corridor | **MEAN (multi-month carrier)** | **1.41** | 1.16 |
| corridor | monthly (incl. eddies) | 1.09 | 1.08 |
| window | mean | 1.21 | 1.14 |

The eddy field is nearly round; the **standing jet/gyre system — the thing
that carries water coherently over multiple months — moves corridor water
1.41× farther east-west than north-south**. The elliptic arm's aspect is
1/1.41 = **0.71**, measured, not styled. (Chris's instinct pointed the right
direction; the measurement sets the magnitude.) A side benefit, pre-measured:
compression pulls slots back inside the window — corridor occupancy 71.6% vs
68.1% circular.

**36 → 34, what 36 was costing.** A golden-angle sequence's bearing gaps come
in at most three sizes (three-distance theorem), and max/min = φ = 1.618
exactly at Fibonacci counts, φ² = 2.618 at every other count. 36 points
therefore carried a blind sector 2.6× its own smallest gap — the exact defect
the spiral exists to remove — where 34 carries none, at two fewer points.
The 36-point arms (#270/#271/#274, ~1 GPU-h sunk) were cancelled for
34-point replacements. 24 stays 24 despite the same φ² gaps: its slot count
(25) is the matched control for both three-rings-wide *and* the elliptic arm,
and that pairing is worth more than gap uniformity.

**Wind and CO₂ — where the other drivers stand.**

- **Wind is already in, direction included.** family3 channels 35–38 are NCEP
  wind stress **τx and τy** (signed components = direction) plus their
  within-month stds, at every pixel. The stencil multiplies this: the model
  now reads the wind field over the whole ellipse, i.e. the *upstream* wind
  that drives the convergence arriving months later. No new plumbing needed —
  every spiral arm already carries it.
- **CO₂ is scoped, deliberately not dispatched** (E-025,
  `ml/plans/E025_forcing.md`). Measured: CO₂ correlates +0.99 with bare time,
  and beyond a smooth clock adds **+0.00043** one-step — for *monthly* rolls
  it is a trend proxy, not a lever, and OHC/EEI is an *output* of the system
  (using it as input is leakage). Where CO₂ genuinely matters is
  multi-decade scenario projection, which is §8 of that plan and worth
  running *after* the geometry question settles, on whatever shape wins here.

**Priority order, by expected information about the programme** (most
promising first, per Chris): **1. elliptic 24@4444×0.71** (#276/#277/#278) —
flow-shaped reach, the design the CFL argument and the anisotropy measurement
jointly pick; **2. circular 24@4444** (#267/#268/#273) — its exact control,
already running; **3. spiral-34@4444** (#279/#280/#281) — density at fixed
reach; **4.** the 222–1000 km factorial (two-rings / wide / narrow /
spiral-13 / spiral-8) — settles bearing-vs-radius at the scales already
known to carry signal.

| arm | runs | shape | slots |
|---|---|---|---|
| **elliptic spiral 24** | #276 / #277 / #278 | golden angle, zonal 111→4444 km, aspect 0.71 | 25 |
| **spiral of 34** | #279 / #280 / #281 | golden angle, 111→4444 km, circular | 35 |
| ~~spiral of 36~~ | ~~#270/#271/#274~~ | cancelled for 34 (φ² blind sector), ~1 GPU-h sunk | — |

---

### SUNFLOWER — elliptic + far-heavy √-ramp, 34 points (Chris, 2026-08-14 evening) · **PRIORITY**

> *"Let's run (and prioritize) an additional experiment ... an eliptic spiral
> but with heavier weight on the outer points. And let's use 34 points in
> total."*

**The principled far-heavy ramp is the literal sunflower.** Vogel's model of
the sunflower head — r ∝ √k at the golden angle — is the unique arrangement
with uniform density per unit *area*, and because area grows quadratically,
uniformity in area IS far-weighting in radius. Encoded as the spiral's 4th
field (ramp exponent on the linear span): `spiral:111-4444-0.71-0.5`. Of 34
points: **26 beyond 2222 km** (geometric ramp: 7), **32 beyond 1000 km**
(geometric: 14). 34 is Fibonacci, so bearings are φ-even.

**Why now.** The early reads attribute the wide arm's −0.0043 to its eight
OUTER points at 1000 km, while density at 222 km bought nothing (#234). The
geometric ramp concentrates points exactly where density was shown not to
pay. This arm moves the weight to where the early evidence says the value is.

**Arms.** #282 (s0, behind the elliptic on `gpu-box-47566393`) · #283 (s1,
running on a box rented for it) · #284 (s2, queued behind s1; the second
rented box, Vast 47726881, stalled provisioning and was not waited for).

**Pre-registered cost, measured before dispatch: corridor occupancy 47.6 %**
(16.2 of 34 slots live) — the steepest structural-zero fraction of any
design, because past 2222 km much of the ellipse is off-window or land. The
arm pays 35 slots of width for ~17 live inputs: E-022's sample-efficiency
risk in its purest form, on the record so the result reads against it either
way. It is still the most *live far-field* coverage of any shape (~16 points
beyond 1000 km).

**Controls.** spiral-34 geometric (#279–#281): same 34 points, same reach,
near-heavy — the ramp is the only difference bar the ellipse. espiral-24
(#276–#278): same ellipse, geometric ramp — the ramp at fixed aspect.
e023r222 and e017 as always.

**Falsifier.** Three sunflower seeds at or below the geometric spiral-34
would say the far field does NOT reward extra weight once reach exists, and
the geometric ramp's near-field coverage was doing real work — sending the
next design back toward a hybrid (near ring + far spiral).

**One near-miss recorded.** The commit carrying this geometry was destroyed
minutes after being written — a `git reset --hard origin/main` chained after
a push that was REFUSED (remote had moved) wiped it, the exact failure root
`CLAUDE.md` §1 documents from 2026-08-07. Recovered from the reflog,
rebased, pushed bare. The rule stands: push bare, read the output, sync in a
separate command — never chain the sync.

---

### THE DESIGNS, DRAWN (Chris: *"Please draw all your designs in the experiment log."*)

**These pictures are generated, not drawn.** `ml/draw_stencils.py` lays a
synthetic all-ocean grid, calls the real `build_stencil` with the real
latitude row, and decodes each neighbour's (dy, dx) back out of the indices
the model would actually gather. So they show the integer rounding onto the
0.25° grid, the 1/cos(φ) zonal stretch, and the half-sector rotation of every
second ring — three things a freehand circle gets wrong — and no shape can
change in `temporal.py` without its drawing changing too. (Same argument as
`rollout_spatial.py --export-mask` writing the globe's AMOC mask rather than
`app.js` tracing a corridor by hand: the second definition is the one that
silently goes stale.) Regenerate with `python3 ml/draw_stencils.py --md`.

![the nine stencil designs, all at one scale](figs/stencil_designs.png)

*The same fourteen as one sheet, **all at a single scale** — which the ASCII
below cannot do (the radial axis is √r, so a 222 km ring and a 4444 km spiral
fit one sheet; bearings are exact and each panel prints its true reach). The
span is the finding: 3×3 reaches 35 km, the champion 222, and the deep
spirals 4436 — a factor of 127 between the first shape tried and the newest. Regenerate with
`python3 ml/draw_stencils.py --svg ml/figs/stencil_designs.svg`.*

Two things to read on each ASCII drawing. The **scale bar**, because at their own
scales the 3×3 that lost by 6.3 seed sd and the 222 km ring that won by 4.4
are the same picture — eight points around a centre, sixty times apart in
width (the figure above is the other half of that: one scale, fourteen panels).
And the **bearing rose** under it (72 characters, 5° each), which is
the quantity the spiral is an argument about: it shows at a glance that three
rings of eight put `||` doubles on eight of their sixteen directions, while a
spiral puts one mark on each of its own.

**3x3 touching (E-022)   [#219-#221]**

```
  9 slots = centre + 8 neighbours  ·  21-35 km (adjacent cells)  ·  8 bearings >=10 deg apart
  the first shape tried: eight cells that TOUCH. LOST by 6.3 seed sd.



           o                   o                   o













           o                   @                   o













           o                   o                   o


  |-----------------| 20 km
  N|......|..........E|.........|.......S|......|..........W|.........|.......N   <- bearings watched, 5 deg/char
  @ = the pixel predicted  ·  lat 40 N, 0.25 deg grid  ·  THE NINE VIEWS ARE NOT TO A COMMON SCALE
```

**13-point (E-022)   [#222-#224]**

```
  13 slots = centre + 12 neighbours  ·  21-56 km (adjacent cells)  ·  8 bearings >=10 deg apart
  5x5 with the outer diagonals trimmed. LOST by 8.1 seed sd.



                               o






                     o         o         o






           o         o         @         o         o






                     o         o         o






                               o


  |----------------------| 50 km
  N|......|..........E|.........|.......S|......|..........W|.........|.......N   <- bearings watched, 5 deg/char
  @ = the pixel predicted  ·  lat 40 N, 0.25 deg grid  ·  THE NINE VIEWS ARE NOT TO A COMMON SCALE
```

**ring of 8 @ 222 km (E-023)   [e023r222]**

```
  9 slots = centre + 8 neighbours  ·  222 km  ·  8 bearings >=10 deg apart
  WON: corridor AUC 0.6043, +4.4 seed sd. The reigning champion.



                              .1.
                      ........   .........
                 .....                    ...
               ...                           ...
            .1.                                 .1
           ..                                     ..
         ...                                        ..
        ..                                           ..
       ..                                             ..
       .                                               ..
      .                                                 .
     ..                                                 ..
     .                                                   .
     .                                                   .
     .1                        @                        1.
     .                                                   .
     .                                                   .
     ..                                                 ..
      .                                                 .
       .                                               ..
       ..                                             ..
        ..                                           ..
          .                                         ..
           ..                                     ..
             1.                                 .1
               ...                           ...
                  ....                   .....
                      ........   .........
                              .1.


  |----------------------| 200 km
  N|.......|.........E|........|........S|.......|.........W|........|........N   <- bearings watched, 5 deg/char
  @ = the pixel predicted  ·  lat 40 N, 0.25 deg grid  ·  THE NINE VIEWS ARE NOT TO A COMMON SCALE
```

**ring of 16 @ 222 km   [#234]**

```
  17 slots = centre + 16 neighbours  ·  222 km  ·  16 bearings >=10 deg apart
  density at ONE radius. n=1, kept as a control, not an arm.



                              .1.
                      ........   .........
                 ....1                   1...
               ...                           ...
            .1.                                 .1
           ..                                     ..
         ...                                        ..
        ..                                           ..
       ..                                             ..
      1.                                               .1
      .                                                 .
     ..                                                 ..
     .                                                   .
     .                                                   .
     .1                        @                        1.
     .                                                   .
     .                                                   .
     ..                                                 ..
      .                                                 .
      1.                                               .1
       ..                                             ..
        ..                                           ..
          .                                         ..
           ..                                     ..
             1.                                 .1
               ...                           ...
                  ...1                   1....
                      ........   .........
                              .1.


  |----------------------| 200 km
  N|...|...|....|....E|...|....|...|....S|...|...|....|....W|...|....|...|....N   <- bearings watched, 5 deg/char
  @ = the pixel predicted  ·  lat 40 N, 0.25 deg grid  ·  THE NINE VIEWS ARE NOT TO A COMMON SCALE
```

**two rings, 8+8 @ 222/555 km   [#237-#239]**

```
  17 slots = centre + 16 neighbours  ·  222/555 km  ·  16 bearings >=10 deg apart
  outer ring rotated half a sector: 16 bearings, not 8 bearings twice.


                           .........
                    .......         .......
                ....2                     2...
              ..                              ....
            ..                                   ...
         ...                                        .
        ..                                           ..
       ..                                              .
      .                                                 .
     2                     ....1....                     2
    .                   ....       ....                  ..
   ..                 .1              .1.                 ..
   .                 ..                 ..                 .
   .                ..                   ..                .
   .                .                     .                .
   .                1          @          1                .
   .                .                     .                .
   .                ..                   ..                .
   .                 ..                 ..                 .
    .                 .1.              1.                 ..
    .                   ....       ....                   .
     2                     ....1....                     2
      .                                                 ..
       .                                               .
        ..                                           ..
          ..                                       ...
            ..                                   ...
              ...                              ..
                 ...2                     2....
                    .......          .....
                           ..........

  |------------------------| 500 km
  N|...|...|....|....E|...|....|...|....S|...|...|....|....W|...|....|...|....N   <- bearings watched, 5 deg/char
  @ = the pixel predicted  ·  lat 40 N, 0.25 deg grid  ·  THE NINE VIEWS ARE NOT TO A COMMON SCALE
```

**three rings, 8+8+8 @ 222/555/1000 km   [#255-#257]**

```
  25 slots = centre + 24 neighbours  ·  222/555/1000 km  ·  16 bearings >=10 deg apart (20 distinct to 1 deg)
  the widest shape yet: 24 points, but only 16 distinct bearings.



                              .3.
                      ........   .........
                 .....                    ...
               ...                           ...
            ...                                 ..
           ..3                                   3..
         ...                   ..                   ..
        ..              .2..... .....2.              ..
       ..            ...              ....            ..
      ..           ...                   ..            ..
      .           ..                       ..           .
     ..          .2         ...1...         2.          ..
     .           .        .1.     .1.        .           .
     .          ..       ..         ..       .           .
     3          .        1     @     1        .          3
     .           .       ..         ..       ..          .
     .           .        .1.     .1.        .           .
     .           .2         ...1...         2.          ..
      .           ..                       ..           .
       .           ...                   ..            ..
       ..            ....              ...            ..
        ..              .2..... .....2.              ..
          .                   ..                    ..
           ..3                                   3..
             ..                                 ..
               ...                            ..
                  ....                   .....
                      ........    ........
                              .3..


  |------------------------| 1000 km
  N|...|...||...|....E|...|...||...|....S|...|...||...|....W|...|...||...|....N   <- bearings watched, 5 deg/char
  @ = the pixel predicted  ·  lat 40 N, 0.25 deg grid  ·  THE NINE VIEWS ARE NOT TO A COMMON SCALE
```

**three rings, 4+4+4 @ 222/555/1000 km   [#275/#259/#260]**

```
  13 slots = centre + 12 neighbours  ·  222/555/1000 km  ·  8 bearings >=10 deg apart
  same reach at half the width. 12 points on 8 bearings, 4 of them twice.



                              .3.
                      ........   .........
                 .....                    ...
               ...                           ...
            ...                                 ..
           ..                                     ..
         ...                   ..                   ..
        ..              ....... .......              ..
       ..            ...              ....            ..
      ..           ..2                   2.            ..
      .           ..                       ..           .
     ..          ..         ...1...         ..          ..
     .           .        ...     ...        .           .
     .          ..       ..         ..       .           .
     3          .        1     @     1        .          3
     .           .       ..         ..       ..          .
     .           .        ...     ...        .           .
     .           ..         ...1...         ..          ..
      .           ..                       ..           .
       .           ..2                   2.            ..
       ..            ....              ...            ..
        ..              ....... .......              ..
          .                   ..                    ..
           ..                                     ..
             ..                                 ..
               ...                            ..
                  ....                   .....
                      ........    ........
                              .3..


  |------------------------| 1000 km
  N|.......|.........E|........|........S|.......|.........W|........|........N   <- bearings watched, 5 deg/char
  @ = the pixel predicted  ·  lat 40 N, 0.25 deg grid  ·  THE NINE VIEWS ARE NOT TO A COMMON SCALE
```

**spiral of 13, 222 -> 1000 km   [#261-#263]**

```
  14 slots = centre + 13 neighbours  ·  222/1000 km (geometric ramp)  ·  13 bearings >=10 deg apart
  the twin of the row above +1 slot: same reach, 13 bearings not 8.

                          ..........
                    ......          .......
                ....                       ....
             ...                               ...
           ..                                     ..
         ..                                         ..
       ..                                             ..
     ...                             9                  .
    ..                                                   .
   ..                                                     .
   .                     6                                 .
  .       b                                                 .
  .                                                     c   ..
 .                             1       4                     .
 .                                                           .
 .                                                           .
 .                     3       @                             .
 .                                                           .
 .                                           7               .
 .                                  2                        .
 ..              8                                          ..
  .                                                         .
   .                         5                             .
   ..                                                     .
    ..                                                   .
      .                                                 .
       ..                              a              ..
         ..                                         ...
          ...                                     ..
             ...                               ...
               d...                        ....
                   .......          .......
                          . ........
  |----------------------------| 1000 km
  N|...|.....|...|...E..|......|...|....S..|...|.....|.....W.|...|.....|......N   <- bearings watched, 5 deg/char
  @ = the pixel predicted  ·  lat 40 N, 0.25 deg grid  ·  THE NINE VIEWS ARE NOT TO A COMMON SCALE
```

**spiral of 8, 111 -> 890 km   [#264-#266]**

```
  9 slots = centre + 8 neighbours  ·  111/890 km (geometric ramp)  ·  8 bearings >=10 deg apart
  the champion's exact width, spent on eight radii instead of one.


                      ...................
                 .....                   ....
              ....                           ....
           ...                                  ...
          ..                                       ..
        ..                                           ..
       .                                              ...
      .                                                 ..
     .                6                                  ..
    .                                                     ..
   .                                                       .
  ..                                                        .
  .                                   4                     .
  .                            1                            .
 .                                                           .
 .                       3     @                             .
 .                                                           .
  .                                2                        .
  .                                                 7       .
  .                                                        ..
   .                                                       .
   ..                        5                            .
    .8                                                   ..
     ..                                                 ..
       .                                              ..
        ..                                           ..
         ...                                       ..
            ..                                  ...
              ....                            ..
                  ....                   .....
                      ...................

  |-------------------------------| 1000 km
  N|.........|.......E..|......|........S..|.........|.....W.|.........|......N   <- bearings watched, 5 deg/char
  @ = the pixel predicted  ·  lat 40 N, 0.25 deg grid  ·  THE NINE VIEWS ARE NOT TO A COMMON SCALE
```

**spiral of 24, 111 -> 4444 km   [#267/#268/#273]**

```
  25 slots = centre + 24 neighbours  ·  111/4444 km (geometric ramp)  ·  21 bearings >=10 deg apart (24 distinct to 1 deg)
  24 bearings AND 4444 km: the first shape that can hold a month of Gulf Stream.



                         ..............
                    .....             ......
                ....                        ...
              ..                               ..
            ..                    m              ..
          ..                                       ..
        ..                                           .
       ..                                             ..
       .                                               ..
      .                                                 .
     .                j                                  .
    .o                        e      h                   .
    .                                                     .
    .                       b 6 9  c                      .
    .                         3@47           k            .
    .                   g    8 5a                         .
    .                        d      f                     .
     .                                                   .
     .                                                   .
      .           l            i                        .
      ..                                               ..
       ..                                             ..
        ...                                          .
          ..                                       ..
           ...                              n    ..
             ....                              ..
                ....                       ....
                    .....             ......
                         .............


  |----------------------------| 5000 km
  N||.|....|.|...|...E|..|...||..|.|...|S..|..|....|.|...|.W|.|.|....|..|..|..N   <- bearings watched, 5 deg/char
  @ = the pixel predicted  ·  lat 40 N, 0.25 deg grid  ·  THE NINE VIEWS ARE NOT TO A COMMON SCALE
```

**spiral of 34, 111 -> 4444 km   [#279-#281]**

```
  35 slots = centre + 34 neighbours  ·  111/4444 km (geometric ramp)  ·  21 bearings >=10 deg apart (34 distinct to 1 deg)
  34 is Fibonacci: phi-even bearings where 36 left a 2.6x blind sector.


                     ....................
                 ....                    .....
              ....                            ...
           ...                                   ..
          ..                                       ..
        ..                                           ..
       .                                u             ...
     ..                                                 ..
    ..     w                                             ..
    .                    r                                ..
   .                                                       .
  ..                            m                           .
  .                                     p                   .
 ..                        j  e   h                      x  ..
 .                    o       6 9                            .
 .                         g b8@7c   k                       .
 .                            d5a f                          .
 ..                       l                                 ..
  .            t               i             s              .
  .                                 n                       .
   .                                                       .
   ..                      q                              ..
    ..                                                   ..
     ..                                                 ..
       .                                               ..
        ..                                           ..
         ...                        v              ..
            .y                                  ...
              ....                            ..
                  ....                   .....
                      ...................

  |-------------------------------| 5000 km
  N||.|.|..|..||.|.|.E|.|.|..||.|..|.|.|S.|.|..||.|..|.|.|.W|.|.|.|..|.|.|.|..N   <- bearings watched, 5 deg/char
  @ = the pixel predicted  ·  lat 40 N, 0.25 deg grid  ·  THE NINE VIEWS ARE NOT TO A COMMON SCALE
```

**ELLIPTIC spiral 24, zonal 111 -> 4444 km, aspect 0.71   [#276-#278]**

```
  25 slots = centre + 24 neighbours  ·  111/4444 km (geometric ramp)  ·  20 bearings >=10 deg apart (24 distinct to 1 deg)
  the flow-shaped arm: corridor mean flow moves 1.41x farther E-W than N-S (measured), so the window has the same proportions.







                        ...............
                  ......               .......
              ....                           ....
           ...                    m              ...
         ...                                        ..
       ..                                             ..
      ..                                               ..
     .                j                                 ..
    .o                        e      h                   ..
    .                       b   9                         .
    .                   g    86@47 c         k            .
    .                        d  a   f                     .
    ..                                                   ..
     ..                                                 ..
      ..          l            i                       ..
       ..                                             ..
         ..                                         ..
           ...                              n    ...
             ....                            ....
                 .......               ......
                        ...............






  |----------------------------| 5000 km
  N|.|.|.....|.|..|..E|.|..||..|..|....|S...|...|...|.|..|.W||.|...|.|....|...N   <- bearings watched, 5 deg/char
  @ = the pixel predicted  ·  lat 40 N, 0.25 deg grid  ·  THE NINE VIEWS ARE NOT TO A COMMON SCALE
```

**SUNFLOWER 34: elliptic + sqrt ramp, 111 -> 4444 km   [#282-#284]**

```
  35 slots = centre + 34 neighbours  ·  111/4444 km (geometric ramp)  ·  22 bearings >=10 deg apart (34 distinct to 1 deg)
  Vogel's sunflower: uniform AREA density puts 26 of 34 points past 2222 km — the far-heavy arm the early reads argue for.







                        ...............
                 .......               .......
              ...    r            m        u ....
           ...                                   ...
         ..                e                        ...
       ..w       j                  9      h          ..
      ..                                           p   ..
     .                   6                               .
    ..   o       b                    4       c          x.
    .                                                     .
    .                   3      @                    k     .
    .        g                     2       7              .
    ..              8                                    ..
     . t                                                ..
      .                      5                f       s..
       ..      l       d            a                 ..
         ..                                         ..
           ...                  i           n    ...
             .y..       q                    ....
                 ......               v......
                       ................






  |-----------------------------| 5000 km
  N|.|..|.|..|.||.||.E|.||.|.|.|..||...|S..|.|..|.|.|.|.||.W||.||..|.|.|..|...N   <- bearings watched, 5 deg/char
  @ = the pixel predicted  ·  lat 40 N, 0.25 deg grid  ·  THE NINE VIEWS ARE NOT TO A COMMON SCALE
```

| shape                                                 | runs           | slots | pts | reach km | bear>=10 | bear~1 | b/pt | gap max/min |
|-------------------------------------------------------|----------------|-------|-----|----------|----------|--------|------|-------------|
| 3x3 touching (E-022)                                  | #219-#221      | 9     | 8   | 21-35    | 8        | 8      | 1.00 | -           |
| 13-point (E-022)                                      | #222-#224      | 13    | 12  | 21-56    | 8        | 8      | 0.67 | -           |
| ring of 8 @ 222 km (E-023)                            | e023r222       | 9     | 8   | 213-224  | 8        | 8      | 1.00 | -           |
| ring of 16 @ 222 km                                   | #234           | 17    | 16  | 213-229  | 16       | 16     | 1.00 | -           |
| two rings, 8+8 @ 222/555 km                           | #237-#239      | 17    | 16  | 213-558  | 16       | 16     | 1.00 | -           |
| three rings, 8+8+8 @ 222/555/1000 km                  | #255-#257      | 25    | 24  | 213-1002 | 16       | 20     | 0.67 | -           |
| three rings, 4+4+4 @ 222/555/1000 km                  | #275/#259/#260 | 13    | 12  | 213-1002 | 8        | 8      | 0.67 | -           |
| spiral of 13, 222 -> 1000 km                          | #261-#263      | 14    | 13  | 223-1003 | 13       | 13     | 1.00 | 1.62        |
| spiral of 8, 111 -> 890 km                            | #264-#266      | 9     | 8   | 111-892  | 8        | 8      | 1.00 | 1.62        |
| spiral of 24, 111 -> 4444 km                          | #267/#268/#273 | 25    | 24  | 111-4436 | 21       | 24     | 0.88 | 2.62        |
| spiral of 34, 111 -> 4444 km                          | #279-#281      | 35    | 34  | 111-4443 | 21       | 34     | 0.62 | 1.62        |
| ELLIPTIC spiral 24, zonal 111 -> 4444 km, aspect 0.71 | #276-#278      | 25    | 24  | 83-4383  | 20       | 24     | 0.83 | 2.62        |
| SUNFLOWER 34: elliptic + sqrt ramp, 111 -> 4444 km    | #282-#284      | 35    | 34  | 83-4339  | 22       | 34     | 0.65 | 1.62        |

---

---

<a id="e-025"></a>
## E-025 · Forcing (CO₂ / energy balance) — SCOPED 2026-08-14, plan written, not dispatched

**Why, from Chris.** *"How should the co2 / the energy balance be modeled as
part of the prediction? Can you thoroughly investigate and make a proposal
that can be implemented by a less capable model?"*

**Full proposal: `ml/plans/E025_forcing.md`.** The scoping measurements
(`ml/measure_forcing_info.py`, no GPU) that shape it:

**1. Over 1982–2024, CO₂ *is* the calendar.** r with a linear time index:
CO₂ **+0.9905**, CO₂ deseasonalised **+0.9946**, AR6 anthropogenic ERF
**+0.9921**. Independent of time: ONI −0.0901, sunspots −0.2519, natural ERF
+0.3851. A model handed CO₂ over this window has been handed `t`, and would
fit the trend, score well on held-out years *inside* the span, and be
indistinguishable from the memorisation E-021 caught — while looking like
physics.

**2. At one step, forcing is worth almost nothing, because the state already
contains it.** Incremental held-out variance over the centre's own 3-month
history: linear time +0.00165, CO₂ +0.00196, **CO₂ beyond a clock +0.00043**,
ONI +0.00005, sunspots +0.00044, everything together +0.00243 — against
E-023's ring at +0.0112. The shuffled-CO₂ control goes negative (−0.00034),
so the instrument is working and the small gain is real temporal alignment.

*Caveat recorded rather than buried:* a pooled linear probe gives a global
scalar one shared coefficient, so it is structurally blind to ENSO's
patterned response. The ONI row means "not measurable this way", not zero,
and the plan's R1 closes it with a per-pixel regression before spending GPU.

**3. The energy balance is an OUTPUT and must not be an input.** Ocean heat
content is a vertical integral of the same temperature field the codec embeds
and the model predicts; feeding it in is feeding a smoothed copy of the
target. Radiative forcing (CO₂, ERF) pushes the ocean and is a legitimate
input; OHC/EEI is the ocean's response and belongs on the validation side.
This distinction is the main structural claim of the proposal.

**Consequence for the design.** The proposal splits forcing into a SLOW
channel (CO₂/ERF, ~0.99 collinear with time, testable **only** on an era
split: train 1982–2010, hold out 2011–2024 entirely) and a FAST channel
(ENSO/volcanic/solar, independent of time, testable on the standard split);
ships a bare linear-time control arm with every slow-channel arm, so that
"we added a clock" is a reportable outcome; and makes the 20-year projection
scenario-conditional (`flat` vs `ssp245`), with the *difference* between
scenarios as the output rather than either trajectory.

**Not dispatched.** ~20 GPU-h, ~$5.4, and it needs its own era-split baseline
arm (nine runs at R3, not six) — worth doing deliberately rather than at the
end of a long night.

---

<a id="e-024"></a>
## E-024 · Is a LARGER array of input pixels better? — MEASURED 2026-08-14, no GPU spent

**Why, from Chris.** *"Can you experiment with more input pixels, adding some
further away? or is there a larger array of pixels that would be particularly
useful?"* — asked after E-023's 8-point ring at 222 km cut one-step error 3.9%.

**Answer: no. Eight points at ~222 km is at or past the optimum, and every
larger array measured is worse.** `ml/measure_shape_info.py`, same instrument
as E-023 (incremental held-out variance explained on top of the centre's own
three-month history), 120 centres, all shapes paired on one sample:

| shape | inputs | on ocean | gain |
|---|---|---|---|
| **ring 8 @ 222 km** | **8** | 0.95 | **+0.0112** |
| ring 4 @ 222 + 4 @ 445 | 8 | 0.94 | +0.0106 |
| ring 8 @ 111 + 8 @ 334 | 16 | 0.95 | +0.0056 |
| ring 16 @ 222 | 16 | 0.95 | +0.0051 |
| ring 8 @ 445 | 8 | 0.91 | +0.0021 |
| ring 8 @ 222 + 8 @ 445 | 16 | 0.93 | +0.0019 |
| ring 8 @ 222 + 8 @ 890 | 16 | 0.90 | −0.0000 |
| ring 8 @ 890 | 8 | 0.87 | −0.0032 |
| ring 8 @ 111 + 8 @ 222 + 8 @ 445 | 24 | 0.94 | −0.0033 |
| ring 16 @ 445 | 16 | 0.91 | −0.0062 |

Three readings, in order of how much they constrain the next experiment:

1. **Density does not help.** Doubling the points on the *same* circle
   (16 @ 222 vs 8 @ 222) *halves* the gain, +0.0051 against +0.0112. Eight
   samples already resolve whatever structure a 222 km circle carries; the
   ninth through sixteenth are interpolations of their neighbours.
2. **Adding a second, farther ring costs more than it brings.** 8 @ 222
   alone beats 8 @ 222 + 8 @ 445 by 6×, and the three-ring shape is
   *negative*. Distance past ~300 km is not a new information channel — the
   445 km ring alone is worth +0.0021 and the 890 km ring is worth less than
   nothing.
3. **Width is the binding constraint, not reach.** The one 16-point shape
   that nearly holds its own (8 @ 111 + 8 @ 334, +0.0056) still loses to half
   its width at one radius. And at FIXED width the split shape
   (4 @ 222 + 4 @ 445, +0.0106) ties 8 @ 222 — so how the budget is spread
   across scales barely matters, while how large the budget is matters a lot,
   in the wrong direction.

**Known bias of the instrument, stated because it cuts toward the answer —
and which turned out to be the whole story for the wide shapes; see the
CORRECTION below.** A pooled ridge pays for every extra column in estimation
variance, and the transformer trains on ~15 M windows rather than 61 k
samples, so the probe should overstate the penalty on wide shapes. Two things stop that from
rescuing a larger array. E-022 measured the same axis in the REAL model —
9 and 13 touching neighbours, both decisively worse than 1 — so the direction
is confirmed where we have both instruments. And the ordering *within* equal
widths (8 @ 222 ≫ 8 @ 445 ≫ 8 @ 890) is a pure information statement that no
dimension penalty explains, since all three have identical width.

### CORRECTION (2026-08-14, same day) — "larger is WORSE" was my estimator, not the ocean

Chris asked the question that caught this: *"I assume a far away point to be
less correlated than a closer one, how can it have lower information gain?"*
The premise was right, and checking it properly broke part of the table above.

**Re-measuring the same shapes with more centres moves every number up, and
moves the wide and far ones up most:**

| shape | 120 centres | 250 centres | 400 centres |
|---|---|---|---|
| ring 8 @ 222 | +0.0112 | +0.0152 | +0.0140 |
| ring 8 @ 445 | +0.0021 | +0.0105 | — |
| ring 8 @ 890 | **−0.0032** | **+0.0022** | — |
| ring 16 @ 222 | +0.0051 | — | **+0.0134** |
| ring 8 @ 222 + 8 @ 445 | +0.0019 | — | **+0.0128** |
| ring 8 @ 111 + 8 @ 222 + 8 @ 445 | −0.0033 | — | **+0.0127** |

At 120 centres doubling the width appeared to *halve* the gain (+0.0051 vs
+0.0112); at 400 centres the two are the same to within 0.0006. The 890 km
ring went from negative to positive. **A ridge pays for every column in
estimation variance, and at 120 centres that cost exceeded the signal in the
wide shapes.** The negative gains were an artefact of my sample size. They
were reported as though they were a property of the ocean, and they were not.

**Why the peak at 222 km is nonetheless real** — `ml/measure_partial_info.py`
decomposes what a neighbour brings into its two competing halves:

| radius | redundancy r(nb_t, ctr_t) | relevance r(nb_t, ctr_{t+1}) | **partial (new AND relevant)** |
|---|---|---|---|
| 28 km | 0.971 | 0.374 | 0.0151 |
| 56 km | 0.941 | 0.370 | 0.0217 |
| 111 km | 0.851 | 0.348 | 0.0271 |
| **222 km** | **0.724** | **0.311** | **0.0280** |
| 445 km | 0.588 | 0.273 | 0.0236 |
| 890 km | 0.398 | 0.216 | 0.0193 |

Chris's premise is confirmed in column one: redundancy falls steeply with
distance, 0.971 → 0.398. But **relevance falls too** — the far pixel knows
less about *this* pixel's next month, 0.374 → 0.216. Usable information is
what survives both, and it peaks in the middle at 222 km. This quantity is
free of any dimension penalty (it is a correlation, not a fit), so the
interior maximum is a property of the data. The RANKING in E-024 stands; the
SIGN and the magnitudes at wide shapes did not.

**What the corrected numbers actually say.** A wider array is not harmful —
it is *redundant*: sixteen points at 222 km buy +0.0134 where eight buy
+0.0140, and twenty-four across three radii buy +0.0127. Double or triple the
input width, no measurable gain. That is still a reason not to spend GPU on a
bigger array, but it is a weaker and different reason than the one first
written here, and the first one was wrong.

### ADDENDUM — information PER POINT, which is the right objective (Chris, 2026-08-14)

*"In theory it would be the more information the better (and redundant does
not hurt). Maybe the actual question is how much information per input point
— that's what we want to maximize."*

Both halves are right. In the infinite-data limit extra inputs cannot hurt,
and `test_zero_weight_equivalence` proves this architecture can represent
"ignore them" exactly. What makes width expensive is not theory but our
training budget: E-022 measured 9 and 13 touching neighbours coming out 6.3
and 8.1 seed sd WORSE than none in the real transformer, which could have
ignored them and did not. So gain per point is the quantity to maximise, and
neither earlier script measured it — they compared whole hand-picked shapes.

`ml/measure_marginal_info.py` does, by greedy forward selection over 6 radii
× 8 bearings, adding one position at a time. **Marginal gain of the k-th
point:**

| k | position chosen | total | **marginal** |
|---|---|---|---|
| 1 | 222 km @ 90° | +0.0065 | **+0.0065** |
| 2 | 333 km @ 225° | +0.0103 | +0.0038 |
| 3 | 222 km @ 0° | +0.0127 | +0.0024 |
| 4 | 890 km @ 180° | +0.0137 | +0.0010 |
| 5–6 | 111 km @ 90°, 270° | +0.0158 | ~+0.0010 |
| 7–8 | 333 km @ 135°, 445 km @ 180° | +0.0165 | +0.0005, +0.0003 |

**And the sets, all scored in one run against one baseline:**

| set | points | gain | **per point** |
|---|---|---|---|
| **greedy top-3** | **3** | +0.0127 | **+0.00423** |
| greedy top-4 | 4 | **+0.0137** | +0.00342 |
| greedy top-6 | 6 | +0.0158 | +0.00263 |
| greedy top-8 | 8 | +0.0165 | +0.00207 |
| uniform ring 8 @ 222 (= E-023's arm) | 8 | +0.0123 | +0.00154 |
| uniform ring 8 @ 333 | 8 | +0.0113 | +0.00141 |
| uniform ring 8 @ 555 | 8 | +0.0061 | +0.00077 |
| **two rings, 222 + 555** | 16 | +0.0087 | +0.00055 |
| all 48 candidates | 48 | −0.0033 | — |

**Three answers.**

1. **How many points?** Three to six. The marginal falls 6.5 → 3.8 → 2.4 →
   1.0 (×10⁻³) and is 0.3 by the eighth. E-023's uniform eight is already
   well past the knee.
2. **Two rings at 222 and 555?** Measured directly: **+0.0087 at sixteen
   points, worse than the single 222 ring's +0.0123 at eight**, and the worst
   per-point number in the table. But the greedy search *does* mix radii
   (222, 333, 222, 890, 111 …) and its eight mixed points beat either uniform
   ring. Mixing scales helps; paying for two full rings to do it does not.
3. **Per point**, the greedy top-3 carries **2.7× the uniform ring's
   information** and 7.7× the two-ring shape's — and **greedy top-4 beats the
   entire 8-point ring (+0.0137 vs +0.0123) with half the inputs.**

**The actionable consequence, and it reverses this entry's recommendation:**
a *narrower* stencil looks better than the one now in the model. A 5-slot
head (centre + 222 @ 90°, 333 @ 225°, 222 @ 0°, 890 @ 180°) has more measured
information than E-023's 9-slot ring at **half the input width**, and E-022
is direct evidence that this model pays for width. That is a training arm
worth dispatching — E-026 — and unlike the wide shapes it is cheap and the
prediction is falsifiable: it should beat e023r222's corridor AUC of 0.6043.

**Caveats on the method, since they bound how far this can be pushed.**
Greedy is not optimal and cannot revise an early pick. The marginals past
k≈6 are near this estimator's own noise. The bearings are absolute compass
directions pooled over every pixel, while the ocean is locally oriented — a
stencil aligned to each pixel's mean current would likely do better again,
and is the natural E-027. And the probe has now been calibrated against real
training twice: it ranks correctly and gets magnitudes wrong in both
directions, so treat +0.0137 vs +0.0123 as an ordering, not a forecast.

**Consequence.** No E-024 training arm is dispatched. The pre-registered
reason to spend GPU would be a shape the probe ranks above the 222 km ring,
and there isn't one; the honest next step for spatial inputs is not a bigger
array but a different KIND of input, and E-025 (forcing) is the untested axis
with an actual physical mechanism. Cost of answering this question: **0 GPU-h**
— roughly 20 minutes of sandbox CPU against the frozen cache.

---

<a id="e-023"></a>
## E-023 · The RING: neighbours far enough away to be new information — PREPARED 2026-08-14, dispatch queued behind R4

**Why, from Chris.** *"Try also a different stencil shape, which is less
correlated and therefore adds new information. For example pixels forming
equidistant points on a circle of radius r (for which r is large enough such
that correlation is lower than 0.99)."*

**The measurement that chose the radius** (`ml/measure_ring_info.py`, frozen
run-62 embeddings, 300 centre pixels, no GPU). For each radius: the mean
correlation between a centre's embedding and its 8 ring neighbours', and the
INCREMENTAL PREDICTIVE INFORMATION — how much held-out one-step residual
variance the ring removes on top of the centre's own last three months, ridge
with weights shared across pixels as stage 2 shares them.

| radius | cells @ eq | corr | ring on ocean | variance gain |
|---|---|---|---|---|
| 27.8 km | 1 | 0.968 | 0.99 | +0.0063 |
| 55.6 km | 2 | 0.936 | 0.97 | +0.0095 |
| 83.4 km | 3 | 0.889 | 0.96 | +0.0125 |
| 111 km | 4 | 0.845 | 0.96 | +0.0150 |
| **167 km** | **6** | **0.783** | **0.94** | **+0.0162** |
| **222 km** | **8** | **0.718** | **0.92** | **+0.0161** |
| 334 km | 12 | 0.642 | 0.90 | +0.0145 |

Truncated there, and said out loud rather than presented as a finished sweep:
445 km and beyond were requested and never returned. The cost per radius grows
with the radius — a far ring's pixels are scattered across the 5.6 GB embedding
memmap, so the gather degrades from page-cache hits to random reads — and the
445 km row ran an order of magnitude longer than the 27.8 km one before it was
killed. The curve already has its maximum and its decline on both sides, which
is what the radius choice needed; the far tail is unmeasured. (The script now
writes after every radius, because this first run kept its rows only in memory
and the numbers above had to be read back out of the log — the same defect,
found the same night, as the evaluator that wrote only at the end.)

Information peaks at **167–222 km** and is **~3× what the touching neighbours
carry**. That is the quantitative form of the mechanism guessed at in E-022:
at one cell the neighbour correlates 0.97 with the centre and is very nearly
a copy of it.

**A prediction of mine died here, and is recorded rather than dropped.** I
expected the 0.99 correlation to be seasonal-cycle inflation, so that
deseasonalising would collapse it and the redundancy would turn out to be an
artefact of the month features. It does not: 0.970 raw vs 0.968 deseasonalised
at one cell. The redundancy is genuine spatial structure.

**Instrument calibration, stated before the arms are dispatched.** At 1 cell
this probe predicts +0.63% and E-022's real 3×3 delivered ~0% on the forecast
ratio (0.19247 vs 0.19216 baseline, n=3 each). So the probe **overstates** what
the transformer realises — unsurprising, since its baseline is a 3-lag linear
AR and the real baseline is a 24-month transformer that has already extracted
much of what a neighbour offers linearly. Read the ring's +1.6% as an upper
bound against a weak baseline, and expect substantially less. Pre-registering
this is the point: if the ring also lands at zero, the honest conclusion is
that neighbours are redundant with the centre's own history at every radius
tested, not that the radius was wrong.

**Hypothesis.** A ring at 222 km carries information the 3×3 does not, and
enough of it survives the nonlinear baseline to move the forecast ratio and/or
the rolled corridor AUC.

**Falsifier.** Three seeds at ring 222 km whose forecast ratio and rolled
corridor AUC both sit inside the e017 stencil-1 seed band. That would close
LOCAL SPATIAL COUPLING for this architecture at monthly cadence at every
reach measured — 1 cell to 12 — and point the programme at what E-021
identified as the other missing ingredient: forcing.

**Design — capacity-matched, radius is the only difference.** `--stencil 9
--ring-km 222` has the same nine slots, the same input width and the same
parameter count (32,338,432) as E-022's `--stencil 9`. Set beside e022s9's
three seeds it is a controlled comparison of DISTANCE, with everything else
including the seeds held fixed. The ring is a circle on the ground (zonal step
× 1/cos φ, per row), because a fixed cell offset spans 27.8 km at the equator
and 9.5 km at 70 °N.

Why 222 km and not 167 km, given their measured gains are equal to 0.0001:
take the less redundant of two equally informative radii (corr 0.718 vs
0.783) — that is Chris's own criterion, and it is the tie-break that does not
depend on a difference smaller than the measurement.

**Arms.** `stencil:9,ring:222` × seeds 0/1/2, 60k steps, U=1, everything else
the e017 recipe. **Controls:** e022s9 (same shape at 1 cell, already run) and
e017 (no neighbours, already run) — no new baseline runs needed.

**INTERIM, two seeds home (2026-08-14 ~09:1xZ) — the ring moves the forecast
objective, which neither E-022 arm did.** Geometry verified in both archives
(`scale.ring_km` 222.0, `stencil` 9, params 32,338,432 — identical to its 3×3
control, so the only difference is distance).

| arm | forecast ratio, mean (sd) | seeds |
|---|---|---|
| e017 (no neighbours, n=3) | 0.19216 (0.00205) | 0.19366 / 0.19300 / 0.18982 |
| e022s9 (3×3 at 1 cell, n=3) | 0.19247 (0.00150) | 0.19290 / 0.19371 / 0.19081 |
| **e023r222 (ring at 222 km, n=3)** | **0.18476 (0.00311)** | **0.18782 / 0.18487 / 0.18160** |

**All three seeds in (#232 landed at 0.18160, the best of them):** −0.0074
against the baseline, i.e. **−3.6 baseline seed sd**, with **every ring seed
below the baseline's lowest** (0.18982) and below every 3×3 seed. A **3.9%
relative cut in one-step z-MSE** where nine touching neighbours bought
nothing. Geometry verified in all three archives (`ring_km` 222.0,
`stencil` 9, params 32,338,432 — identical to the 3×3 control).

The nowcast k-fold reads 0.449 against the baseline's 0.485, i.e. slightly
lower — but that probe's seed sd is ~0.12, so at three seeds it says nothing
in either direction and is quoted only to avoid the appearance of picking the
metric that agreed.

**Both of my calibration expectations were wrong, in opposite directions**,
and that is worth more than the number itself. I pre-registered that the
linear probe *overstates* what the transformer realises, because at 1 cell it
predicted +0.63% and the real 3×3 delivered ~0%. At 222 km it predicted +1.6%
and the real ring delivered ~3.0% — it *understated* by about a factor of two.
So the probe ranked the radii correctly (its central claim, and the one the
radius choice rested on) and was wrong about magnitude at both ends. A probe
that gets the ordering right and the size wrong is still the right instrument
for choosing a radius and the wrong one for predicting an effect.

**This does NOT decide E-023.** The pre-registered primary is the rolled
corridor AUC, and E-022 is the standing proof that the two can disagree: its
3×3 arm was flat on this exact forecast metric and 6.3 seed sd WORSE once
rolled. One-step error and twelve-step behaviour are different questions.
Seed 2 (#232) and the gated evaluation follow.

### RESULT (2026-08-14 ~12:0xZ) — the ring WINS, on the pre-registered primary metric

The gate passed **exactly** for the third independent time (AUC 0.643, bands
0.470/0.375/0.492 in #228, #229 and #233 alike), so the four arms below are
comparable across runs: the evaluator is deterministic given a head, and it
has now reproduced #217 on three separate boxes.

| arm (3 seeds) | **corridor AUC** | vs baseline | window AUC | amp h12 | AMOC h1-3/h4-6/h7-12 | long ho r |
|---|---|---|---|---|---|---|
| stencil 1 (e017) | 0.5837 [0.580–0.589] | — | 0.6193 | 0.742 | 0.458/0.354/0.464 | 0.401 |
| 3×3 at 1 cell (e022s9) | 0.5537 | −6.3 sd | 0.5987 | 0.731 | 0.466/0.369/0.451 | 0.373 |
| 13-point (e022s13) | 0.5453 | −8.1 sd | 0.5897 | 0.753 | 0.472/0.324/0.366 | 0.361 |
| **ring 222 km (e023r222)** | **0.6043 [0.599–0.608]** | **+4.4 sd** | **0.6257** | **0.765** | 0.460/**0.400**/**0.492** | 0.354 |

**The pre-registered falsifier is cleared.** The bar was 0.5837 + 3 × 0.0047
= **0.5978**, and all three ring seeds (0.599, 0.606, 0.608) sit above it.
This is the first arm in the programme to beat the no-neighbour baseline on
the primary metric — after unroll (E-010/E-020), local coupling (E-022) and
bigger arrays (E-024) each failed.

**It is the same intervention that failed at one cell.** Identical
architecture, identical parameter count (32,338,432), identical seeds; the
eight neighbours are simply 222 km away instead of 28. Moving them is worth
**+0.021 corridor AUC**, where leaving them adjacent cost −0.030.

**Two secondary readings support the mechanism rather than merely agreeing.**
Amplitude retention at h=12 rises 0.742 → 0.765: the rolled field damps less,
which is what a coupling term that carries real information should do, and
the opposite of what a smoothing input does. And the AMOC transport bands
improve exactly where a ~200 km/month coupling scale predicts — h4-6 0.354 →
0.400 and h7-12 0.464 → 0.492, while h1-3 is flat (0.458 → 0.460), because
at one to three months the centre's own history already carries the answer.

**Recorded against the result:** the long hindcast's held-out r is *lower*
(0.354 vs 0.401) and its three seeds span 0.281–0.393, which at that spread
says nothing either way. It is quoted because it is the one secondary metric
that does not agree.

**Physics.** 222 km per month is ~8.5 cm/s — the interior and deep-western-
boundary flow scale from the E-022 plan's own table, not the Gulf Stream's
100–200 cells/month. The experiment that worked is the one aimed at the
speed the model can actually resolve at monthly cadence.

**Cost.** Three training arms (75/118/74 min) + one 4-head evaluation
(144 min) = **6.9 GPU-h, ~$1.9**, plus ~40 min of sandbox CPU for the radius
measurement that chose 222 km before any GPU ran.

**DISPATCHED 2026-08-14 ~05:15Z: #230 / #231 / #232** (seeds 0/1/2,
`stencil:9,ring:222`, 60k steps, job_timeout 400) — seeds 0 and 2 pinned to
gpu-box-42005419, seed 1 to gpu-box-40623952. They QUEUE behind the R4
evaluation (#228, #229) already running on both boxes and start as those
finish. Implementation and tests landed first (13 green, commit 20c967c9c);
workflow token `ring:` wired with the block re-measured at 20,670 of 21,000
chars (commit 36318d756).

---

<a id="e-022"></a>
## E-022 · Spatial coupling: predict a pixel from its NEIGHBOURHOOD — R1 smoke DISPATCHED 2026-08-13

**Why, from Chris.** *"Implement predictions that depend on the neighboring
pixels … either 9 pixels (3×3) or a 5×5 … as an optimization leave out the
second diagonal, so you'd have something more circular: 2 in each direction
but only one on the diagonal. As a baseline, the eval needs to involve
rolling forward all pixels that contribute to the AMOC current (from the
Gulf of Mexico to northern Europe, and back)."* Follows directly from
E-021/E-021b's diagnosis: the per-pixel head has zero cross-pixel coupling,
so nothing can advect — rolls decay to a seasonal limit cycle.

**Full pre-registered design: `ml/plans/E022_spatial_coupling.md`** (commit
59175146a) — hypothesis, falsifiers, the physics table, settled decisions,
dispatch plan R0–R5. Summary of the wager:

**Hypothesis.** Feeding each pixel's stencil of embeddings (S ∈ {9, 13},
missing neighbours zero-filled and flagged in the static context) lets
stage 2 represent local transport/diffusion → higher rolled corridor skill,
slower amplitude decay, better held-out-year hindcast tracking.

**Primary metric & falsifier.** Rolled **AUC over the AMOC corridor**
(data-derived: train-month mean cur_speed ≥ p75 ∪ RAPID section), h=1..12,
3 seeds/arm, vs the e017 stencil-1 band. One correction to the plan's §1
made at implementation, before any spatial number existed: the quoted
0.643–0.645 band is rollout.py's `horizon_auc` = mean MSSS-vs-CLIMATOLOGY
(the damped mean is 0.619) — the plan wrote "AUC(msss_damped)"; the gate
targets what #217 actually measured. Falsifier: no spatial arm's 3-seed
mean beats 0.645 + 3×(pooled seed sd) on the corridor → local monthly
coupling closes. **Pre-registered physics caveat:** one roll step = one
month; the stencil reaches 1–2 cells/month, Gulf Stream advection moves
100–200 — a null closes LOCAL MONTHLY coupling only, not spatial coupling
at daily cadence or with a global operator.

**Controls.** The three published e017 heads re-rolled by the NEW evaluator
(`ml/rollout_spatial.py`, full-window roll of all 84,405 px), behind a
FATAL validation gate: e017_u1_s0 must reproduce #217's gate-subset AUC
0.643 and truefit bands 0.470/0.375/0.492 within ±0.01 or nothing spatial
gets scored. The zero-weight-equivalence test (tests/test_e022_stencil.py)
proves a stencil model can represent the stencil-1 model exactly, so a
REGRESSION would be evidence about training dynamics, not capacity.

**Scale (per R2 arm).** 576/8 trunk as e017 (~32M + the wider `inp`:
9×64+2 or 13×64+2 input columns), batch 256, 60,000 steps, U=1, K=24,
~26–28M train windows. Six arms: stencil {9,13} × seeds {0,1,2}.

**R0 (sandbox, 2026-08-13):** 8/8 stencil unit tests green (zero-weight
equivalence exact at 1e-6; planted-advection toy: stencil-9 MSE < 0.6× of
stencil-1; stencil-1 rebuilds legacy shapes strict=True). Evaluator toy
e2e green incl. the gate refusal (tests/test_rollout_spatial.py). Code:
bbf73c44e (temporal.py stencil), 841e74080 (rollout_spatial.py),
15743e4f9 (workflow wiring).

**R1 smoke (this dispatch):** `stencil:9,seed:0`, 6,000 steps, pinned
gpu-box-42005419 (47483091 start is resource-queued at its host). Measures
the real step time (expectation 180–400 ms/step from the S× gather; if
>400 the gather moves to a prefetch thread before R2), checks loss falls
and temporal.json lands in the archive. The head is thrown away — 6k steps
is not an arm. Also the first run of the re-sectioned FC probe
(probe_kfold `fc` now reads the Florida Straits lon (−80.5,−78.5), 7 cells
— the old basin-wide section diluted it 50:1 and read a false null).

**R1 smoke (#221, 2026-08-13 20:20Z → green in ~35 min).** Everything the
smoke was for, answered: stencil-9 model confirmed live from the record
(params 32,338,432 = 32.038M + exactly the 300,096 stencil input columns);
**~56 ms/step** — the plan's 180–400 ms estimate was pessimistic (the Z
tensor is RAM-resident; torch's advanced-index gather is cheap), so no
prefetch thread is needed and a 60k arm is ~1 h, not 3–4; loss fell
2.00→1.79 by step 1260 on the planned warmup; temporal.json in the
archive (trainer alive at the end); rapid_probe_kfold 0.579 at 6k steps —
a smoke number, quoted only as "alive". **Side result, FC probe
re-section:** first measurement of `fc` on the Florida Straits section
(lon −80.5..−78.5, 7 cells): **r_kfold 0.320 [0.213, 0.428]** (n=490,
lowpass18 0.337) vs wind-only **0.121 [0.013, 0.226]**. The old
basin-wide section's −0.014 "null" was measurement error (50:1 dilution),
as suspected — the codec does carry Florida Current signal.

**R2 DISPATCHED 2026-08-13 ~20:5xZ:** stencil:9 seeds 0/1/2 = **#222 /
#223 / #224**, queued on gpu-box-42005419 (60k steps, job_timeout 400,
plan-22N.json published by dispatch_run). gpu-box-40623952 (47483091) is
resource-unavailable at its Vast host — start queued; the stencil:13 trio
dispatched once it woke: **#225 / #226 / #227** on gpu-box-40623952
(same 60k/expdecay recipe, ~20 min behind the s9 trio).

### RESULT (2026-08-14 07:5xZ) — spatial coupling does not help, and at this cadence it HURTS

Both evaluations passed the fatal validation gate **exactly**: the new
full-window evaluator reproduced #217's stencil-1 numbers to three decimals in
each run (AUC 0.643, bands 0.470/0.375/0.492, `gate.pass: true`), independently
on two boxes with TF32 on. Every number below is therefore from a verified
instrument, not a new one that happens to agree with itself.

Three seeds per arm, all rolled by the same code over all 84,405 window pixels:

| metric | stencil 1 (e017) | 3×3 (e022s9) | 13-pt (e022s13) |
|---|---|---|---|
| **corridor AUC** (primary) | **0.5837** [0.580–0.589] | **0.5537** [0.551–0.558] | **0.5453** [0.539–0.555] |
| window AUC | 0.6193 [0.617–0.622] | 0.5987 [0.595–0.602] | 0.5897 [0.586–0.597] |
| gate-subset AUC | 0.6440 [0.643–0.645] | 0.6250 [0.619–0.629] | 0.6147 [0.603–0.625] |
| amp ratio at h=12 (corridor) | 0.742 | 0.731 | 0.753 |
| AMOC truefit h1-3 / h4-6 / h7-12 | 0.458 / 0.354 / 0.464 | 0.466 / 0.369 / 0.451 | 0.472 / 0.324 / 0.366 |
| long-hindcast r, held-out months | 0.401 | 0.373 | 0.361 |
| nowcast k-fold (temporal.json) | 0.485 | 0.500 | 0.510 |
| forecast ratio vs persistence | 0.19216 | 0.19247 | 0.19091 |

**The falsifier fired, in the direction nobody pre-registered.** The rule was
"no arm beating 0.5837 + 3 sd (= 0.5978) closes local monthly coupling". The
baseline's own seed sd on the corridor is 0.0047, and the spatial arms land
**−0.030 (−6.3 sd)** and **−0.038 (−8.1 sd)** BELOW it. Not a null: a
consistent, large regression, monotone in the amount of neighbourhood
supplied — more neighbours, less skill — on every rolled scope (corridor,
window, gate subset) and on the long hindcast.

**This is evidence about optimisation, not about capacity, and the experiment
was built so that claim is not a rationalisation.** `test_zero_weight_equivalence`
proves a stencil-9 or -13 model can represent the stencil-1 model exactly (a
zeroed neighbour block, `allclose` at 1e-6), so every one of these arms could
have matched the baseline by ignoring its extra inputs and did not. What the
extra inputs cost is sample efficiency: E-023's measurement puts the neighbour
correlation at **0.97 at one cell**, so a 3×3 adds ~9× the input columns and
almost no information — collinear predictors, diluted gradient signal, same
step budget.

**The one place the neighbourhood is not worse is the short AMOC band**: h1-3
truefit runs 0.458 → 0.466 → 0.472, ordered the *other* way. Three seeds on a
240-month probe with sd ≈ 0.12 cannot support that as a finding; it is noted
as the only counter-current in the table, not claimed.

**Pre-registered physics caveat, restated because the result needs it.** One
roll step is one month; a 3×3 reaches 1 cell/month and the 13-point 2, against
Gulf Stream advection of 100–200 cells/month. This experiment therefore closes
**LOCAL coupling at MONTHLY cadence** for this architecture. It says nothing
about daily cadence, nor about a global operator (attention over the whole
field), nor — the E-023 follow-up — about neighbours far enough away to carry
information the centre does not already have.

**Cost.** 9 dispatches, **15.5 GPU-hours, ~$4.19**: R1 smoke 30 min; six 60k
arms 73–105 min each (533 min); two evaluations 212 + 156 min. The evaluation
itself was 5.71 GPU-h across 10 head-evals, a flat **34.3 min per head** (714
roll steps × 84,405 pixels). Zero dead dispatches — the toy end-to-end test and
the fatal gate both did their jobs.

**What it changes.** The unroll axis closed at E-010/E-020; local spatial
coupling closes here. Of E-021's two named missing ingredients — spatial
coupling and forcing — one is now measured and rejected at this cadence and
reach. E-023 (already running) tests the remaining spatial hypothesis: that the
useful neighbour is not the adjacent one but the distant one.

**Interim, first arm home (#222, stencil:9 seed 0, 2026-08-13 ~22:40Z).**
Green, temporal.json archived, `scale.stencil` 9, 32,338,432 params, 60,000
steps, 28,857,960 train windows. Two numbers, and the quiet one is the
informative one:

| run | stencil | nowcast k-fold | z-MSE | forecast ratio vs persistence |
|---|---|---|---|---|
| e017 s0 (#208) | 1 | 0.497 [0.389, 0.599] | 0.6080 | 0.1937 |
| e017 s1 (#209) | 1 | 0.497 [0.387, 0.597] | 0.6034 | 0.1930 |
| e017 s2 (#210) | 1 | 0.462 [0.341, 0.568] | 0.5978 | 0.1898 |
| **e022s9 s0 (#222)** | **9** | **0.437 [0.336, 0.529]** | **0.6056** | **0.1929** |
| **e022s9 s1 (#223)** | **9** | **0.547 [0.426, 0.659]** | **0.6056** | **0.1937** |
| **e022s9 s2 (#224)** | **9** | **0.516 [0.411, 0.607]** | **0.6009** | **0.1908** |
| **e022s13 s0 (#225)** | **13** | **0.541 [0.397, 0.660]** | **0.6065** | **0.1932** |
| **e022s13 s1 (#226)** | **13** | **0.486 [0.372, 0.605]** | **0.6013** | **0.1923** |

The nowcast probe (0.437 vs 0.462–0.497) is BELOW the baseline seeds but at
n = 1 on an instrument with seed sd ≈ 0.12 (E-010) that means nothing yet —
do not quote it. The **forecast ratio is the one that can speak at n = 1**:
that objective reproduces to sd ≈ 0.0017 across seeds, and 0.1929 lands dead
centre of the stencil-1 spread 0.1898–0.1937. Nine times the input columns,
and the one-step prediction of z is unchanged to within a fifth of the
baseline's own seed range.

**The 3×3 trio is complete (01:55Z) and both training-time metrics are
null.**

| arm | nowcast k-fold, mean [range] | forecast ratio, mean (sd) |
|---|---|---|
| e017 (stencil 1, n=3) | 0.485 [0.462–0.497] | 0.19216 (0.00205) |
| e022s9 (3×3, n=3) | 0.500 [0.437–0.547] | 0.19247 (0.00150) |
| e022s13 (13-pt, n=2) | 0.514 [0.486–0.541] | 0.19276 (0.00061) |

Forecast: the 3×3 is +0.0003 on the baseline — **0.15 of the baseline's own
seed sd**, and again in the worse direction. Nowcast: +0.015 against a
per-seed sd of ~0.12 (sem ~0.07), a fifth of one standard error. Neither
training-time metric can see the neighbourhood, at either reach. Worth noting
for the earlier n=1 claim: the spatial arms' own ratio sd (0.0015 and 0.0006)
is no LARGER than the baseline's 0.0021, so reading one seed on that metric
was not resting on an unmeasured assumption about this architecture.

**Three arms in (00:25Z), and the forecast null is no longer n = 1.** Baseline
mean ratio **0.1922** (sd 0.0021, n = 3); spatial mean **0.1933** (sd 0.0004,
n = 3 across BOTH stencils). The gap is +0.0011 — half the baseline's own seed
sd, and in the direction of very slightly WORSE. Every spatial arm lands inside
the stencil-1 range [0.1898, 0.1937], and the 13-point (double the reach, 13×
the input columns) is indistinguishable from the 3×3. On the objective these
models actually optimise, neighbourhood inputs buy nothing at either reach.

The nowcast k-fold behaved exactly as its noise floor says it must: #222's
0.437 looked low, #223 came back 0.547 on the identical recipe — a 0.11 swing
between seeds of one arm, against a baseline mean of 0.485. Anyone quoting
either number alone would have "found" an effect in both directions. This is
E-010's lesson arriving on schedule; the probe is not usable at n < 3.

*Post-hoc interpretation, flagged as post-hoc — this was noticed AFTER the
number arrived, not predicted before it:* E-021b measured the spatial
correlation of z on this very cache at **r = 0.99 at one cell** and 0.88 at
five. If a neighbour's embedding is 0.99-correlated with the centre's, a 3×3
stencil is not new information, it is the same information nine times — the
per-pixel model was already, in effect, reading a locally smoothed field.
That would predict exactly this null on the forecast objective, and it is a
mechanism the plan's physics caveat (reach 1–2 cells/month vs advection
100–200) did not name.

**This does NOT decide E-022.** The pre-registered primary metric is the
ROLLED corridor AUC at h=1..12 (R4), and identical one-step error is
compatible with different rollout behaviour — error STRUCTURE, not error
size, is what governs amplitude decay over twelve steps. Seeds 1–2 (#223,
#224) and the 13-point trio (#225–#227) are still running; the verdict waits
for the gated eval.

---

<a id="e-020"></a>
## E-021 · The 20-year fan: long-horizon AMOC projection with ensembles — #219, 2026-08-13 08:02Z → RESULT same day

**Why, from Chris.** *"Could we try to now run repeated predictions on 20
years of AMOC and observe stats on the different possible outcomes?"* and
then *"plot the current for the next 20 years as well as predicted vs
measured for the past 20 years."*

**Process note, recorded against myself.** Rule 1 of this file says the
entry is written AT DISPATCH. This one was not — the design went straight
from chat into `ml/project_amoc.py` and a `project:` window token, and the
entry is being written after the numbers landed. The hypothesis below is
reconstructed from the script's docstring and the dispatch `doc` field,
both of which predate the result; but the rule exists precisely so that
claim doesn't have to be taken on trust, and it was broken.

**Design.** Six published 32M heads (`e017_u1_s0..2`, `e020_u4_s0..2`)
rolled autoregressively on frozen run-62 embeddings over the RAPID
section only — exact, not an approximation, because the head attends over
TIME per pixel with a static spatial identity and has no cross-pixel
coupling for a full-grid roll to add. Two ensemble families, 12 members
each: **ic** perturbs only the initial 24-month context, **sde** injects
noise at every rolled month; both scaled by each head's OWN measured
teacher-forced one-step residual, so the spread is a measurement rather
than a chosen number. Two starts: `future` (context ends 2024-12) and
`2004-12`, whose 240 rolled months land on 2005–2024 where RAPID exists.
Read-out is rollout.py's truefit ridge (val-tail r 0.606). Cost: ~45 min
on one 4090, one dispatch, no dead runs.

**Hypothesis.** The fan quantifies "possible outcomes"; the 2004 start
calibrates its width before the future fan is believed.

### RESULT — the fan is not a forecast, for two independent reasons

**1 · The hindcast tracking is largely MEMORISATION.** The 2004 roll
receives no data after 2004-12 and still follows RAPID at **r = +0.78**
(ic) / +0.55 (sde) across twenty years. That is impossible for a genuine
forecast, so it was treated as a bug and chased:

| test | result | verdict |
|---|---|---|
| lag-12 autocorrelation of the median | 0.26 | not a seasonal loop |
| r(median, its own first-12 loop) | −0.07 | not a seasonal loop |
| r detrended | 0.726 | not a trend artefact |
| r of first differences | 0.66 | month-specific, not low-frequency |
| year-block-shuffled null | mean 0.00, 95th 0.16 | not chance |
| **r(FUTURE roll 2025–44, truth 2005–24)** | **+0.05** | **specific to the initial condition** |
| agreement across 6 independent heads | ±0.01 | not one seed's quirk |

Split by what the codec actually trained on: **r = +0.78 on trained
months vs +0.42 on the three held-out years** (2009 +0.53, 2017 +0.09,
2023 +0.67). A model trained on 2005–2024 replays those years when handed
a matching starting state; the initial condition acts as a key.

**2 · The ensemble is UNDER-DISPERSED — but see E-021b, part of this was
my method, not the model.** As first measured: pooled 90% band ≈ **0.9 Sv**
against **10.3 Sv** of observed spread, containing **19%** of observations
where a calibrated band contains ~90%. I attributed all of it to the
rolled dynamics being **contractive** (it is — the future tail settles
onto a seasonal limit cycle near −0.8 Sv, lag-12 autocorrelation 0.99).
Chris then asked whether spatial correlation reaches the roll through the
embeddings, which exposed that the perturbations were white per-pixel
noise against a strongly correlated field. **E-021b corrects the number to
~5×, not ~10×.** Read that entry before quoting either.

**What it points at.** The model has **no forcing input** — no surface
heat flux, no freshwater, and no wind after the context ends. Unforced,
the only thing a dissipative learned dynamics can do is decay to
climatology, which is exactly what both panels show. This is the first
measurement that ranks the data backlog: surface fluxes ahead of further
capacity work.

### E-021b — the perturbations were white; the field is not. #220, same day

**Why, from Chris.** *"Two pixel embeddings depend on other pixels near
them. So maybe the spatial correlation comes via the embeddings. No?"*
Yes — and measuring it turned a reported model failure partly into a
method failure of mine.

**Measured on the run-62 cache, along the 26.5°N section:**

| separation | 1 cell | 5 | 20 | 80 | 150 |
|---|---|---|---|---|---|
| r of z | 0.986 | 0.875 | 0.680 | 0.345 | 0.247 |

e-folding near 18°. Pixels 20° apart share no input cells, so this is
inherited from the ocean, not manufactured by the 3×3 encoder patch. The
consequence for the read-out is the whole point: the section mean of 265
pixels has an **effective N of 2.5**, measured as var(section mean) /
mean(per-pixel var). White per-pixel noise therefore averages down by
√265 ≈ 16× where structured error averages down by √2.5 ≈ 1.6× — so
E-021's ensemble was narrowed ~10× by its own noise model.

**Design.** Identical to E-021 except the perturbation: instead of
synthesising white noise at the measured per-dimension scale, resample the
head's OWN one-step residual FIELDS ([S, dz] snapshots kept during the
residual pass). That carries the true spatial and cross-dimension
covariance exactly, with nothing fitted and no free parameter.
`--noise white` retains the original as the control, so exactly one thing
changes. Pre-registered falsifier: if coverage stayed near 19% the
under-dispersion was the model's and E-021 stood unchanged; if it rose
toward 90% the calibration claim was mine and had to be retracted.

**RESULT — neither pole; the honest answer is in between.**

| | white (E-021) | field (E-021b) |
|---|---|---|
| hindcast **sde** coverage of the 90% band | 19.5% | **42.4%** |
| hindcast **sde** mean band width | 1.11 Sv | **2.08 Sv** |
| future **sde** mean band width | 1.00 Sv | **2.12 Sv** (2.1×) |
| hindcast **ic** coverage | 13.9% | 14.7% |
| future **ic** band width | 1.15 Sv | 1.18 Sv (1.0×) |

**My ~10× estimate was itself an over-claim, and the reason is
instructive.** The √N argument governs the noise AT INJECTION; the
contraction then decides how much survives. For **sde**, which
re-injects every month, the fan reaches an injection-versus-decay balance
and correlated structure buys **2.1×** — real, and less than predicted.
For **ic**, which injects once, the structure of the kick is irrelevant by
month 12 because the dynamics has absorbed it either way: **1.0×**, no
change at all. That contrast is itself the cleanest evidence that the
contraction is real and dominant.

**So the corrected verdict.** The fan is still under-dispersed — 2.1 Sv
against 10.1 Sv observed, **~5× too narrow, not ~10×** — and roughly half
of what E-021 charged to the model was mine. The remaining factor is the
model's, and the diagnosis stands: a contractive per-pixel dynamics with
no forcing input cannot manufacture variance.

**Unchanged by this.** The memorisation finding concerns the MEDIAN, not
the spread, and both runs agree on it (field: r +0.72 trained / +0.34
held-out; white: +0.55 / +0.16 — the same story, and the corrected run
reads slightly *more* memorised, not less). Cost: ~20 min on one 4090.

**The lesson, general enough to keep.** A perturbation is a claim about
the error's *structure*, not only its size, and a spatially-averaged
read-out is exactly where an unstructured claim is punished by √N. Where
the real errors are available as fields, resample them; do not synthesise
noise and hope the covariance did not matter.

**Consequences.**
- **No hindcast over the training era may be quoted as skill** — here or
  in the paper. This entry is the reason why.
- The honest forecast evidence remains the h=1..12 rollout numbers
  against damped persistence (E-017 ROLLOUT / #211).
- The figure ships the negative result on its face:
  `ml/figs/amoc_projection.html` (built by `ml/plot_projection.py`), with
  the held-out years shaded and both r values in the subtitle.

**Falsifier for the memorisation claim, if anyone wants to retest it:** a
head trained with 2005–2024 fully excluded should hindcast those years at
the held-out r (~0.4) rather than the trained r (~0.78). We have not run
that; the future-roll control is what stands in for it.

---

## E-020 · U=4 at the 32M trunk: does unroll's probe gain survive capacity? — DISPATCHED 2026-08-12 ~18:10Z

**Why, from Chris.** *"It would be nice to have our current best: best
decoder, best large stage 2, with U=1 and U=4 done till the morning."*
U=1 at 576/8 exists (E-017, three seeds). U=4 has only ever been run at
the 1.8M trunk, where it is a genuine nowcast-probe specialist: **+0.09
k-fold at every seed (E-013b), +28% one-step forecast cost, no horizon
gain**. Nobody knows whether that composes with capacity — and the
stakes changed today: U=1's head probe SATURATED at ~0.49 across
10.7M→32M while the state/label ceiling sits at 0.63 (E-019's
decomposition). If unroll's compression mechanism still buys +0.09 at
32M, the head closes most of the remaining gap to the ceiling.

**Design.** Identical to E-017's arms except `unroll:4`: 576/8 =
**32,038,336 params · batch 256 · 60,000 steps · ~28M windows** (the
trainer's scale block is authoritative), expdecay no-taper, tensor
`adcbe700`, frozen run-62 codec, seeds {0, 1, 2}, pinned two-and-one
across the boxes, job_timeout 600 (U=4 ≈ 2.5–3× stage-2 wall time).

**Pre-registered questions.** (1) Does the nowcast probe move ABOVE
E-017's 0.462–0.497 band — toward the 0.63 ceiling — with U=4-typical
seed-tightness? (2) Is the forecast cost still ~+28% (predict z-ratio
≈ 0.24–0.25 vs U=1's 0.190–0.194)? (3) Held-out val curve and amp: does
full-depth BPTT at 32M stay stable? Rollout at horizon is a follow-up
eval (the #211 protocol, and through the retrained decoder), not part
of these runs.

**Falsifier.** Probe stays inside U=1's band → the U=4 gain is a
small-trunk phenomenon and the axis closes at scale; the morning
package is then U=1 + best decoder, and the remaining nowcast gap
belongs to the read-out/labels, not the trunk.

**Result** (2026-08-13, #212/#213/#214, one per seed). The falsifier
fired — but only the third seed said so, which is the E-010 lesson
enforcing itself:

| seed | run | k-fold (deseas) | 95% CI | z-ratio (t+1) |
|---|---|---|---|---|
| 0 | #212 | **0.556** | [0.463, 0.642] | 0.2602 |
| 1 | #213 | **0.521** | [0.414, 0.623] | 0.2542 |
| 2 | #214 | **0.443** | [0.327, 0.550] | 0.2531 |

After two seeds (0.556, 0.521 — both above E-017 U=1's 0.462–0.497
band) this read as "unroll composes with capacity." Seed 2 landed
BELOW the band. Spread 0.113 at n=3, means 0.507 (U=4) vs ~0.480
(U=1): a +0.027 difference under a ~0.12 seed sd is **no detectable
nowcast effect**, the same verdict E-010 returned at the 1.8M trunk
— where the +0.09 had at least been seed-consistent. At 32M it is not
even that. (2) answered as predicted: z-ratio 0.253–0.260 vs U=1's
0.190–0.194, ~+33% teacher-forced h=1 tax — the objective works as
designed. (3) training stable at full-depth BPTT, amp healthy, no
val divergence.

**So the morning package's nowcast column is settled** — U=1 and U=4
are interchangeable there, and the remaining gap to the 0.63 ceiling
belongs to the read-out/labels axis (E-019's decomposition), not the
trunk or the unroll objective. Cost: 3 runs × ~3.3–4 h ≈ 10.6 GPU·h
on two 4090s, zero dead dispatches.

### E-020 ROLLOUT (#217, 2026-08-13) — the horizon column closes too, and against U=4

The one place U=4 could still have earned its ~+33% teacher-forced tax
was the ROLLED regime the unroll objective actually optimises, which
teacher-forced z-ratio is structurally blind to. All six 32M heads
rolled on identical points, three seeds each:

| metric (h=1..12) | U=1 (E-017) | U=4 (E-020) |
|---|---|---|
| **AUC(msss_damped)** | 0.643 / 0.644 / 0.645 — **mean 0.644, sd 0.0008** | 0.556 / 0.568 / 0.563 — **mean 0.562, sd 0.005** |
| msss_damped at h=12 | +0.576 | +0.469 |
| amplitude retention at h=12 | 0.805 | 0.710 |
| AMOC truefit h1–3 | 0.458 [0.450–0.470] | 0.500 [0.463–0.529] |
| AMOC truefit h4–6 | 0.353 [0.339–0.375] | 0.377 [0.345–0.421] |
| AMOC truefit h7–12 | 0.463 [0.425–0.492] | 0.445 [0.387–0.526] |

**U=4 is decisively WORSE where it should have been better.** Field
skill against damped persistence falls 0.644 → 0.562 — a gap ~16 seed
standard deviations wide, i.e. the cleanest signal in the whole unroll
programme — and amplitude retention falls with it at every horizon
(0.883 → 0.795 at h=1, 0.805 → 0.710 at h=12): the unrolled objective
makes the model MORE smoothing, not less. The AMOC transport bands
differ by 0.02–0.04 with seed spreads of 0.02–0.07, i.e. nothing.

**The U axis is now closed at both trunk sizes and on every axis we
measure**: nowcast probe (null, E-020), teacher-forced forecast (−33%),
rolled field skill (−13%), amplitude (−12%), transport at horizon
(null). E-010 reached the same verdict at 1.8M; capacity does not
rescue it. `--unroll` should default to 1 and the knob is done.

Cost: #217 ran 4h51m of the documented CPU-bound `rollout.py` burn
(~$1.35) — the `model.to(_dev)` fix is still unlanded and is now the
single cheapest infrastructure win outstanding.

---

<a id="e-019"></a>
## E-019 · COPY RECONSTRUCTION: how much does the codec's round trip lose? — audit dispatched 2026-08-12 ~10:20Z, RESULT ~10:35Z

**Why, from Chris.** *"We take some input. We encode it. We decode it, and
then we look how far away it is, and it should be almost identical."* His
framing: the 0.631 read-out ceiling is encoder-derived — the bottleneck
throws information away before any predictor sees it — and a system aiming
at the best possible AMOC prediction should not be paying an avoidable
representation tax. The decoder programme starts by MEASURING that tax.

**What has never been measured.** The codec trains on MASKED reconstruction
(hidden channels weight 1.0, visible 0.1, Huber); its final train loss_rec
≈ 0.09 is a masked-dominated mix. Nobody has ever scored the
full-visibility round trip — encode exactly what stage 2 and the probes
consume (mask = none), decode all 39 channels at offset 0, compare — and
that is the identity Chris is asking about.

**Design (E-019a, the audit).** `ml/recon_eval.py`: the 26.5°N section
(265 px × 516 months), encode via `temporal.embed_everything` itself (the
production code path, on a 3-row latitude slab whose streaming
standardization is verified against the exact in-RAM recipe on two full
channels before scoring), decode all channels, per-channel r and RMSE in
standardized units (RMSE² ≈ fraction of channel variance lost). Split
three ways along the codec's own blocked holdout: train · held-out months
(2009/2017/2023) · held-out lon block (−45,−25). Pooled section-mean
scores reported separately (the transport ridge reads pooled features).
Runs on sandbox CPU — no GPU spend. **Scale: 40,693,xxx-param codec
(run-62 `f3_anchor41M`), eval-only, 136,740 encoder forwards, 0 training
steps.**

**Pre-registered questions.** (1) How far from identity is the round trip
— mean r, and which channels lose most? (2) Do the transport-carrying
channels (ssh, deep rg_t/rg_s) reconstruct better or worse than average?
(3) Does fidelity hold on held-out months and the held-out lon block
(compression that generalises), or collapse there (memorisation)?

**Falsifier.** If full-visibility reconstruction already reads r ≈ 0.99
everywhere, the bottleneck is NOT materially lossy at the section, the
0.631 ceiling is a read-out/label-noise story rather than an
encoder-lossiness story, and decoder-capacity work is the wrong lever —
E-006/E-018 remain the levers. Large per-channel losses instead name
exactly what d_z / decoder capacity / loss weighting should recover.

### RESULT (sandbox CPU, `ml/runs/recon_audit/recon_eval.json`) — the round trip is good, NOT identical, and the loss concentrates exactly where the transport lives

Mean r **0.975** per-pixel / **0.979** pooled over 39 channels (train
split). Both exactness checks passed on the real tensor (streaming vs
in-RAM recipe, max |Δ| 1.9e-06); section resolved to the production 265
pixels.

**(1) Distance from identity, by channel group** (per-pixel variance
lost = rmse², train split):

| group | variance lost | worst members |
|---|---|---|
| winds (tau_*) | 1.0–1.2% | — |
| ssh | 1.8% | — |
| upper-ocean rg_t/rg_s (10–700 dbar) | 2.1–2.3% | — |
| deep rg_s (900–1900) | 3.5% | rg_s1900 r=0.952 |
| **deep rg_t (900–1900)** | **6.9%** | **rg_t1900 r=0.938, rg_t1700 r=0.941** |

**(2) The transport carriers answer in two directions**: ssh
reconstructs best of the ocean channels (r 0.990), but the DEEP
TEMPERATURE channels reconstruct worst — 6.9% of variance lost,
3× the upper ocean — and deep salinity is second-worst. RAPID's
mid-ocean transport is thermal wind: the vertical integral of the
density gradient, i.e. precisely deep T and S. The bottleneck
preferentially discards the channels the transport integral weights.
Pooling does NOT cancel it (pooled deep-T r ≈ 0.95): the probe's
input genuinely lacks deep-density detail. This is Chris's
"encoder-derived prediction uncertainty", measured and localised.

**(3) It generalises**: held-out months and the held-out lon block read
within noise of train (e.g. rg_t1900: 0.938/0.940/0.936) — genuine
compression, not memorisation. The audit's falsifier did NOT fire: the
round trip is not ≈0.99, so decoder work has a real target.

**E-019b design (next), REVISED per Chris's co-training question into two
stages that separate the two possible culprits.** The audit is a LOWER
BOUND on what z contains — it can under-read z if the ~1.3M decoder is
too small to EXPRESS what the encoder kept. So:

- **E-019b1 — decoder-only, encoder FROZEN.** Retrain just a bigger
  decoder (4 hidden × d_dec 1536, ~7M params) against run-62's frozen
  embeddings. z does not change, so every probe and stage-2 head is
  untouched by construction; the question is purely *"was the deep-T
  variance in z all along, unexpressed?"* Recovers → the audit was
  decoder-limited, z is richer than measured, and rollout field skill
  improves for free (chan_skill decodes through query()). Does not
  recover → the information is genuinely absent from z, and only b2
  can put it there. Nearly free: trains from the published Z cache
  with zero encoder forwards.
- **E-019b2 — full co-trained retrain** (the codec has ALWAYS been
  co-trained end-to-end — one AdamW over encoder+decoder, loss
  l_rec + 0.5·l_nei — so this is a retrain with different weights, not
  a first co-training): d_z=64 held fixed, per-channel loss weights
  upweighting deep rg_t/rg_s, visible-channel weight raised from 0.1,
  the bigger decoder. Changes z; scored by this audit + `probe_kfold`
  vs 0.631. Falsifier: recon improves but the probe does not move →
  the lost variance was not transport-readable, and the ceiling is a
  label/read-out story after all. d_z=128 is the rung after, only if
  b2 moves the probe.

Normalization check (Chris asked whether channel amplitudes could be
driving the deficit): every dynamic channel is deseasonalised per-pixel
then standardised to UNIT VARIANCE per channel (train-years,
non-holdout-lon stats), so the loss and the audit see equalised
amplitudes, and r is scale-free besides. Verified at the section:
rmse ≈ √(1−r²) for the deep channels (rg_t1900: 0.332 vs 0.347), i.e.
section-local variance ≈ the global unit — the deep-T deficit is not an
amplitude artefact. Residual normalization-adjacent contributors worth
remembering: Huber's transition sits at 1.0 in these units (tail-heavy
channels get extremes down-weighted), and global-per-channel
standardisation leaves spatial variance structure inside a channel (the
optimizer samples basin-wide; a channel whose variance lives away from
26.5°N earns little gradient there).

### E-019b1 RESULT (sandbox CPU sweep, `ml/recon_decoder.py`, `ml/runs/recon_decoder/`) — the deep-ocean information was in z all along

Multi-output decoders (z → hidden^L → 39) trained on the published
f16 Z cache (verified against a local f32 re-encode before training),
6M identical pairs from train months × non-holdout lons, scored by the
E-019a audit. Deep-T variance lost (train / held-out months):

| decoder | params | deep-T lost | held-out months |
|---|---|---|---|
| production (E-019a) | 1.3M | 6.9% | 7.0% |
| retrained 768×2 | **0.67M** | 2.2% | 2.4% |
| retrained 1536×3 | 4.9M | 1.4% | 1.9% |
| retrained 3072×3 (⅓ the budget) | 19.2M | 1.6% | 2.2% |

A decoder HALF the production size recovers most of the "lost" deep-T
— so E-019a's deficit was TRAINING EMPHASIS (masked-dominated
objective, visible weight 0.1, equal channel weights, neighbour
duties), not encoder loss and not even decoder capacity. Held-out
months (excluded from codec training, decoder training, and the
stats) and the held-out lon block read within noise of train
throughout: real compression, not memorisation. On a samples-seen
axis the 19.2M curve dominates the 4.9M everywhere they overlap with
NO train-val gap anywhere — budget-limited, not data- or z-limited;
extended runs are in flight. b2 (encoder retrain) is NOT needed for
fidelity.

**Weights DURABLE (2026-08-13, #216/#218 — the `dectrain:` window
mode).** The retrained weights were lost to sandbox container
restarts three times (~2h restart cadence vs a ~65-min CPU train), so
decoder retraining moved to the boxes: `window:
dectrain:<h>x<l>@<steps>@<pairs>` (scripts/dectrain_run.sh; the run
extracts X from the box's sha-verified tensor, uses the box's own Z
cache — verified against a local f32 re-encode — and publishes
weights + audit JSON to model-checkpoints-v1 with an
assert-the-effect re-list). Two findings worth the log: (1) the 4090
does the whole 4000-step optimisation in **15 seconds** (the sandbox
CPU took ~43 min), and the box trajectory matched the sandbox
step-for-step to 4 decimals — same pairs, same optimum, CPU/CUDA
agreeing; (2) the extended budget (12k steps, 12M pairs) reproduced
the lost run exactly: best_val 0.00713, deep-T rmse² train 0.85% /
**held-out months 1.90%** / held-out lons 1.43%. That decoder is now
`dec1536x3s0__decoder.{pt,json}` on model-checkpoints-v1 —
replace-don't-accumulate, so the name always carries the current best
1536×3 seed-0. En route, the Probes `run:` block hit GitHub's
21,000-char dispatch-time expression ceiling (422s EVERY dispatch of
the workflow, the 26th-input failure shape; #215 is the phantom
parse-failure marker) — window-token bodies now live in scripts/.

### The ceiling decomposition (`ml/probe_state_ceiling.py`) — 0.631 is a STATE/LABEL ceiling, not a representation tax

Chris asked what the "decoder-induced ceiling" becomes. The decoder
was never in the 0.631 chain, so the answer is a decomposition: the
exact probe_kfold protocol over feature sets bracketing the pipeline
(protocol checks reproduced the published numbers: pooled z 0.627 vs
0.631 published; wind 0.568 exactly):

| features | ridge | MLP |
|---|---|---|
| pooled z (64f) | 0.627 [0.503, 0.735] | 0.611 |
| pooled TRUE fields (39f, decoder=identity) | 0.631 [0.496, 0.746] | 0.530 |
| 5-segment z (320f) | 0.646 [0.539, 0.741] | — |
| 5-segment TRUE (195f) | 0.653 [0.549, 0.741] | — |

**z matches the true uncompressed state at every read-out** (and beats
it under the MLP — compression aids small-sample learnability). So the
0.631 is what THIS monthly-mean section state yields to a 240-label
read-out — not an encoder tax, not a decoder tax. Nonlinearity does
not help at this n; zonal structure helps mildly (+0.02, inside the
CIs). Consequences: E-018 (layer sweep) is predicted null on the probe
and downgraded; E-019b2 unnecessary for the probe; the levers that
remain on the nowcast axis are LABELS (longer series — the Florida
Current cable reaches back to 1982) and read-out structure under
sample constraints; the levers on the FORECAST axis are unchanged
(capacity at horizon, E-006).

---

<a id="e-017"></a>
## E-017 · The second capacity rung: 768/8 — DISPATCHED 2026-08-11 ~22:45Z

**Why.** E-015's first seed cut the forecast error ratio 0.39 → **0.25**
and read k-fold 0.504 — the largest single-change improvement in the
programme's stage-2 history, on the axis E-008 predicted (parameters, not
compute). Chris: *"should we increase capacity further?"* First dispatch
was 768/8 = 56.9 M (5.3×, #205–#207, cancelled unstarted); **revised per
Chris to a 3× rung** — *"I would try params 3x instead of 5x, also to
avoid running out of memory"* — **d_model 576 / 8 layers = 32,038,336
parameters (2.99×)**, U=1, same schedule, same tensor, seeds {0, 1, 2}.
Tighter rungs also draw a better scaling curve.

**Risk, and why now is the right time to take it:** 26 M train windows
are far fewer effective samples (spatial correlation), so a 57 M-param
head can memorise — and tonight's runs are the FIRST to log the held-out
z-MSE curve, the amplitude ratio and the in-training probe trend every
~600 steps. Divergence between train and val curves will be visible live
on the status page, not discovered at eval.

**Pre-registered questions.** (1) Does the z-ratio keep falling (0.39 →
0.25 → ?), and does the val curve confirm it generalises? (2) Does the
probe follow (E-015 s0's 0.504 vs the 1.8 M band)? (3) Where does the
scaling bend — if 5.3× more parameters buys much less than the last
5.9× did, the data or the embeddings are the binding constraint and
E-006 (the objective) is the next lever, not another rung.

**Scale: 32,038,336 parameters · batch 256 · 60,000 steps · ≈26.1 M
windows.** Dispatch note: E-015's remaining seeds (#194/#195) had not
landed when this was queued — accepted risk (~$2 of GPU) to use the
overnight boxes; if they contradict seed 0, this entry records it.

### RESULT (#208–#210, landed overnight, closed 07:50Z 08-12) — the forecast keeps scaling; the transport probe has SATURATED

| seed | k-fold | z-ratio |
|---|---|---|
| 0 | 0.497 [0.389, 0.599] | 0.1937 |
| 1 | 0.497 [0.387, 0.597] | 0.1930 |
| 2 | 0.462 [0.341, 0.568] | 0.1898 |

**Q1 — yes, and cleanly**: z-ratio **0.190–0.194** (seed spread 0.004),
val curves healthy throughout (the first arms with live monitoring: val
tracked ≈0.24 of persistence from a quarter of the way in, amplitude
0.93, no divergence). The scaling ladder now reads **0.39 → 0.25 → 0.19**
(1.8 M → 10.7 M → 32 M).

**Q2 — no. The probe did NOT follow the second rung**: mean **0.485**
(sd 0.020) against the 10.7 M trio's **0.486** (sd 0.024) — identical.
Capacity lifted the transport nowcast exactly once (0.434 → 0.486, the
first rung) and then saturated, ~0.15 below the codec's own 0.631.

**Q3 — the bend, answered**: the second rung bought 24% relative forecast
improvement where the first bought 36% — bending but alive for the
FORECAST. For TRANSPORT, capacity is done: this is E-008's lesson at the
next level ("the thing more buys is not the thing we want"), and it moves
the AMOC bottleneck decisively to the OBJECTIVE and the EMBEDDINGS —
E-006's territory. A third rung is justified only for field-forecast
goals, not for the probe. *(Revised by the rollout below: the nowcast
probe saturated, but AMOC-at-HORIZON did not.)*

### ROLLOUT (#211, closed ~09:55Z 08-12) — at horizon, capacity has NOT saturated; the 32 M trunk is the programme's best AMOC forecaster

Instrument: `rollout.py` staggered starts into 3 holdout years, K=24,
h=1..12, MSSS per Goddard 2013 vs clim/persistence/damped-AR1; AMOC via
the truefit band ridge on rolled section states (the noisiest instrument
in the ladder — but read here at 3 seeds per rung). Baselines measured
on identical points in #200: 1.8 M trunk AUC(msss_damped) 0.27–0.29,
truefit ensemble bands 0.41/0.16/0.23.

| rung | AUC(msss_damped), 3 seeds | truefit h1-3 | h4-6 | h7-12 |
|---|---|---|---|---|
| 1.8 M (#200) | 0.27–0.29 | 0.41 | 0.16 | 0.23 (ens) |
| 10.7 M E-015 | 0.499 / 0.510 / 0.518 | 0.466 | 0.348 | 0.398 (seed means) |
| 32 M E-017 | **0.643 / 0.644 / 0.645** | 0.458 | 0.353 | **0.463** |

**Field skill at horizon scales without any visible bend** — 0.28 →
0.51 → 0.64, and the 32 M seeds agree to 0.002. The skill CURVE also
changes shape: seed-mean msss_clim runs 0.760 (h=1) → 0.586 (h=12),
nearly flat beyond h≈6, with amplitude held at 0.81–0.88 (the small
trunks smoothed toward climatology; 32 M keeps variance out to a year).
ACC 0.87 → 0.77 across the year.

**AMOC at 7–12 months lead roughly DOUBLES over the small trunk**
(0.23 → 0.46), and the E-017 triple is tight (0.492/0.425/0.473, sd
0.034) where E-015's spread is wide (0.314/0.568/0.313 — seed 1 a high
outlier throughout). So the Q2 "saturation" is a NOWCAST-only story:
what capacity stopped buying is current-month transport read-out; what
it kept buying is the dynamics that carry transport information months
ahead. For the main objective — predicting AMOC, not nowcasting it —
the capacity axis is still open, and a third rung is back on the table
alongside E-006/E-018. (rolledfit bands again read broken — 0.16–0.21
at h4+ — consistent with the established overfit; truefit remains the
instrument.)

---

<a id="e-016"></a>
## E-016 · SAMPLED unroll depth: P(U=1..4) = 0.5/0.25/0.125/0.125 — DISPATCHED 2026-08-11 ~21:30Z

**Why, from Chris.** *"Assuming U_max=4, probabilistically set U to 1 (50%
of the time), to 2 (25%), to 3 + 4 (12.5% of the time). Can you try
this?"* The measured trade it targets: fixed U=4 pays **+28% one-step
z-MSE at every seed** while buying the nowcast probe **+0.09 at every
seed** (E-013b). Sampling the depth per training step spends half the
steps on the pure one-step map and reaches full depth only 12.5% of the
time — E[extra forwards] = 0.875 vs 3.

**Design.** `unroll:4, uprobs:0.5-0.25-0.125-0.125`, otherwise pinned to
the E-012 arms (60k, expdecay no-taper, tensor `adcbe700`, frozen run-62
codec), seeds {0, 1, 2}. One depth draw per STEP (whole batch shares it);
empty probs reduce bit-identically to fixed depth (toy-asserted). Labels
`u4p_s<N>` — never ensembled with fixed-U arms. **Scale: 1,822,144
params · batch 256 · 60,000 steps · ≈26.0 M windows.**

**Pre-registered questions.** (1) Does the nowcast-probe gain survive
(k-fold near U=4's 0.502–0.551 band rather than U=1's 0.363–0.493)?
(2) Does the one-step cost shrink (z-ratio near 0.39 rather than 0.50)?
(3) Rollout field skill: does it keep U=1's AUC instead of U=4's deficit?

**The interesting outcome space:** all three yes = strictly dominates
both parents and becomes the default objective. Probe follows the
EXPECTED unroll (landing mid-band) = the compression is dose-dependent.
Probe stays at U=1 levels = the gain needs the full-depth gradient every
step, and the axis closes with fixed U=4 as a transport-nowcast
specialist.

### RESULT (#202/#203/#204, closed 23:30Z) — half the cost, none of the stability; the compression mechanism needs full depth

| seed | k-fold | z-ratio |
|---|---|---|
| 0 | 0.250 [0.085, 0.394] | 0.4396 |
| 1 | 0.527 [0.402, 0.628] | 0.4409 |
| 2 | 0.488 [0.366, 0.615] | 0.4333 |

**Q2 first, because it answered cleanly: the one-step cost is halved and
dose-proportional** — z-ratio 0.433–0.441 (E[U] = 1.875 lands almost
exactly midway between U=1's 0.39 and U=4's 0.50), seed-stable to 0.008.
**Q1: the probe gain does NOT survive as a reliable effect.** Mean 0.422
(below even U=1's 0.434), but the real finding is the SPREAD: sd 0.150
(range 0.277) against fixed U=4's 0.025 — sampling destroys precisely
the seed-stability that made fixed U=4's probe result credible. The
variance-compression mechanism evidently needs the full-depth gradient
at every step; applied stochastically it becomes a coin flip. Q3 (field
AUC) left unmeasured — with Q1 answered this way, a rolleval of these
heads buys nothing; skipped, recorded as such.

**E-016 CLOSED: half-measures pay half the cost deterministically for an
unreliable benefit.** The unroll menu is now fully mapped: U=1 = best
forecaster; fixed U=4 = nowcast-probe specialist (+0.09, tight seeds,
+28% forecast cost); sampled = neither. If the nowcast probe ever
matters enough to pay for, pay full price.

**First arm (#202, seed 0, 22:2xZ).** k-fold **0.250** [0.085, 0.394] —
NOT in U=4's 0.50–0.55 band; at/below the U=1 band. z-ratio **0.4396** —
roughly the dose-proportional midpoint (E[U]=1.875) between U=1's 0.39
and U=4's 0.50. One seed; if it holds, the probe gain needs the
full-depth gradient every step and half-measures pay half the cost for
none of the benefit. Its `scale` block also just earned its keep: it
reads **batch 256** — temporal.py's default; the workflow's `--batch
512` goes to the CODEC trainer — catching this very log's backfilled
"512" as the hand-carried error rule 6 exists to prevent. Corrected
throughout.

**Result.** *pending — seeds 1/2 (#203/#204).*

---

<a id="e-015"></a>
## E-015 · Stage-2 WIDTH: the parameter-bottleneck arm — PREPARED 2026-08-11, dispatch after E-013

**Why, from Chris.** *"In your next experiments, please consider trying out a
larger stage 2 model (u=1) as well."* This is the arm E-008 explicitly left
open: 33× compute moved the forecast objective steadily and the AMOC probe
not at all, which closed the compute bottleneck and said nothing about
CAPACITY. Nobody has ever trained stage 2 at more than d_model 192 / 4
layers.

**Design.** d_model 384 / 6 layers, from scratch, U=1, 60,000 steps,
`sched:expdecay --lr-cooldown-frac 0` (horizon-free, hot endpoint —
extendable without a warm restart), same codec, same tensor `adcbe700`,
seeds {0, 1, 2}. Everything else pinned to E-012's U=1 arms, which are
the direct comparison set. **Scale: 10,732,096 parameters (5.89× the
trunk's 1,822,144) · batch 256 · 60,000 steps · ~26.1 M train windows.**

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

### RESULT (#193–#195, closed 07:50Z 08-12) — capacity was the forecast bottleneck, and it helps the probe once

| seed | k-fold | z-ratio |
|---|---|---|
| 0 | 0.504 [0.389, 0.611] | 0.2498 |
| 1 | 0.458 [0.363, 0.543] | 0.2500 |
| 2 | 0.495 [0.377, 0.596] | 0.2455 |

**All three pre-registered questions answer YES-with-shape.** (1) The
probe moves up: mean **0.486** vs the 1.8 M trio's 0.434, with U=4-like
tight seeds (sd 0.024 vs 0.066) — and WITHOUT unroll's +28% forecast
tax, which makes width strictly preferable to fixed U=4 as a probe
carrier (U=4's 0.524 edge costs a fifth of the forecast). (2) Answered
by #211: AUC(msss_damped) 0.499/0.510/0.518 vs the 1.8 M trunk's
0.27–0.29 on identical points, truefit bands 0.466/0.348/0.398 (seed
means; seed 1 a high outlier) vs 0.41/0.16/0.23 — width moves every
horizon metric; full table and the capacity-at-horizon verdict in
E-017's ROLLOUT section. (3) z-ratio **0.245–0.250**
(replicating to 4 decimals across seeds) against the 1.8 M plateau of
0.39 — the largest stage-2 improvement in the log, on the axis E-008
predicted. E-017 (one rung further) then showed the probe SATURATES here
while the forecast keeps scaling — see that entry.

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
**Scale: 1,859,200 parameters (trunk 1,822,144 + three 12,352-param
heads) · batch 256 · 60,000 steps · 26,073,420 train windows (measured,
#191 — the h=12 reach guard trims the plain pool slightly).**
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

**First arm landed (#185, seed 0, 19:0xZ) — the direct heads WORK in
z-space, and the trunk pays at t+1.** `z_direct` beats frozen-z
persistence at every trained horizon: ratios **0.556 (h=3), 0.501 (h=6),
0.651 (h=12)** — and the direct MSE barely grows with horizon (2.47 →
2.61 → 2.66), the signature of a model relaxing toward the conditional
mean rather than compounding. The cost: the shared trunk's teacher-forced
z-ratio is **0.6705** against the plain trio's ~0.389 — the three-horizon
term pulls hard. Head k-fold 0.418 [0.328, 0.517] sits inside the U=1
band (nowcast unharmed), split 0.614, tensor `adcbe700` ✓. Whether the
direct predictions beat the PLAIN heads' iterated rollout in channel
space and on AMOC — the question that matters — is the rolleval after
all three seeds land.

### RESULT (#201, ~21:45Z) — FALSIFIED, by its own pre-registered falsifier

The paired rollout eval (three direct heads beside the three plain U=1
trunks), and the direct heads lose on every count that matters:

- **Direct beats its own iterated path by nothing**: paired
  `delta_msss_clim_vs_iterated` spans −0.014 to +0.028, mean ≈ +0.003 —
  zero, on identical cells.
- **The direct predictions are MORE smoothed, not less**: amp_ratio
  0.54–0.56 at every trained horizon and every seed. The near-flat
  z-MSE across horizons was the tell — the heads learned the
  conditional mean, which beats FROZEN persistence easily (the z_direct
  ratios 0.50–0.66 were real) and predicts no better than iterating.
- **The three-horizon term damaged the trunk**: the direct arms'
  iterated rollout collapses to AUC 0.094–0.109 against the plain
  trunks' 0.266–0.291 — the t+1 cost (z-ratio 0.65–0.67 vs 0.39)
  compounds exactly as E-011 taught.
- Direct-AMOC reads: tiny n, noise, nothing.

The falsifier written at dispatch — "paired deltas ≤ 0 with amp_ratio no
better → the horizon loss is not an artifact of iteration; it is in the
embeddings or the predictability itself, and no readout change will buy
it back" — fired almost verbatim. **The iterated rollout of a well-trained
one-step model remains the best forecaster at every horizon**, and the
h=3/6/12 skill it shows (E-011's positive finding) is not improved by
predicting those horizons directly. E-014 closed; `--direct` stays in the
tree as a documented dead end. The one caveat: this tested direct heads
SHARING a trunk with the t+1 objective at equal weight — a
separate-trunk or down-weighted variant is untested, and nothing above
motivates spending GPU on it.

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

**E-013b — the two U=4 seeds, dispatched 18:40Z as #196/#197** (seeds
1/2, same recipe as #173, pinned to the new third box
`gpu-box-46045353`; plans published). When they land: rolleval all three
U=4 heads beside the U=1 trio for the 3-vs-3 deep-band answer.

**E-013b RESULT, probe level (#198/#199 landed 20:5xZ) — U=4 beats U=1
at EVERY seed on the transport probe at convergence:**

| seed | U=4 k-fold | U=1 k-fold | U=4 z-ratio |
|---|---|---|---|
| 0 | 0.519 [0.375, 0.636] | 0.363 | 0.4953 |
| 1 | 0.502 [0.387, 0.593] | 0.446 | 0.5095 |
| 2 | 0.551 [0.406, 0.677] | 0.493 | 0.5003 |

Means 0.524 vs 0.434; U=4's seed sd **0.025** against U=1's 0.066 —
E-010's variance-compression observation (F = 9.5, then "not
established") is replicating at convergence, and the one-step cost
(+28%) is invariant as ever. At 6k this comparison was a null; at 60k it
is a clean 3-vs-3 ordering. The unroll axis, closed for FORECASTING,
reopens as a candidate TRANSPORT-READING objective: the compressed
representations carry the probe better. #200 (rolleval over all six 60k
heads, dispatched 20:55Z on the idle third box) asks whether the
deep-band rollout advantage (E-013: 0.278/0.335 from one arm) holds
3-vs-3.

### E-013b CLOSED (#200, 21:15Z) — the nowcast advantage is real; the deep-band flip was seed-0 luck

The 3-vs-3 rollout (truefit probe on rolled states, per head):

| head | AUC | h1–3 | h4–6 | h7–12 |
|---|---|---|---|---|
| u1 s0/s1/s2 | .282/.291/.266 | .414/.365/.437 | .182/.082/.201 | .177/.183/.300 |
| u4 s0/s1/s2 | .236/.232/.254 | .394/.310/.277 | .278/.202/.083 | .335/.139/**−.029** |

3-seed ensembles: u1 0.414/0.156/0.227 vs u4 0.347/0.202/0.162. **The
E-013 deep-band flip does not replicate** — u4_s0's 0.335 at h7–12 sits
next to +0.139 and −0.029 from its sibling seeds; seed means put U=1
ahead at h1–3 and h7–12 and the field AUC stays U=1's at every seed.
E-013's "direction, not result" caveat did exactly its job.

**Final unroll verdict, in full, at convergence:** U=4 is (a) a better
NOWCAST-probe carrier at every seed (0.524 vs 0.434, sd 0.025 vs 0.066 —
the one finding that survived), (b) a worse forecaster at every horizon
and every seed, (c) not better at AMOC-at-horizon. The probe-at-horizon
instrument itself is the noisiest read in the suite (u4's h7–12 spans
0.364 across seeds on identical rollout points — the field metrics span
0.022) — treat single-seed band results as direction only, always.

**#196/#197 died on a lemon box (18:50Z).** #196's trainer hit
`torch.AcceleratorError: CUDA error: unspecified launch failure` six
minutes into stage 2 — a host/GPU fault, not code (#185, same commit,
trained happily on box 2 at that moment; its anomalously fast 19.7 ms/st
was likely the same sick GPU) — and #197's deps install then failed on
the same host. **A rigor tell worth keeping: #196 CONCLUDED "success"
with a dead trainer**, because temporal.py runs backgrounded behind
best-effort guards; the missing `temporal.json` in the probe archive is
the reliable signal, never the run's colour. Box 47486012 destroyed;
replacement 47487801 (`gpu-box-42005419`, Ukraine, 100 GB) rented,
labelled, and the seeds re-dispatched as **#198/#199** (final).

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
**Scale (rule 6, backfilled): 1,822,144 parameters · batch 256 · 60,000
steps · ≈26.1 M train windows.**

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

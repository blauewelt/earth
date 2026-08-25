# Session handoff — 2026-08-25 day session (written ~16:25Z, mirrored to the repo ~16:45Z)

For the next agent, whichever account it runs in. Read `ml/CLAUDE.md` first
(it governs), then diff reality against the standing-expectations ledger —
`claude/expectations.md` in the claude.ai project (§5.27; its ~15:45Z state
is summarised in "In flight" below, but the project copy is the live one
where a project exists). A CONCURRENT working session ran E-049 today and
wrote its own handoff (`ml/handoffs/2026-08-25-roadB-session.md`) — the two
sessions' work interleaves in `ml/EXPERIMENTS.md`; do not re-harvest each
other's runs. Resuming from a clone alone (no claude.ai project): the road-B
handoff's resume-from-clone section covers bootstrap; the credential docs
exist only in the project and must be re-created by hand in a new one
(github-access, vast-access, gcp-tpu-access, copernicus-marine-access,
huggingface-access).

## What this session did (chronological, all pushed to main)

1. **Embed-cache partial publish + resume** (2ab7e83, 0af27a2): Z chunks now
   publish DURING the embed every ~10 min with a manifest; an interrupted
   pass resumes at the first missing row on any box; the pull loop's
   64-chunk cap (would have truncated daily-scale Zs) raised to 676. Chris's
   two standing rules written as `ml/CLAUDE.md` §5.26 (partial progress to a
   SAFE location) and §5.27 (the expectations ledger, diffed every
   check-in). The hourly fleet-health check's prompt now includes the §5.27
   diff.
2. **Morning triage**: the clean pentad Z published 06:37Z
   (`Z_071ef7e181_37e146384b`, from #463's embed). #463 (E-045.1 K=144
   decisive rung)'s trainer died silently at step 4,000 → cancelled,
   re-dispatched as **#478** (job_timeout 1400 — 900 was the kill bracket;
   #475 was a first copy cancelled for exactly that). #468 (E-046-HEAD seed
   0, cancelled 08:30Z by the operator on its divergence falsifier) → re-run
   as **#477 seed 1**. **#476** re-queued the held K=48 rung. Queue-stall
   remedy (cancel+rerun, <90 s) used once more on #478.
3. **E-046 VERDICT LANDED** (#477 green 14:26Z): 20k one-step ratio
   **0.4394** — below A9's 0.4916 and the continuous pair's 0.5056/0.5045.
   Training THROUGH the lattice wins; the price is a violently spiky
   optimization (8 gmax>10⁴ excursions at seed 1, all absorbed; seed 0 died
   of its 7th). Paper updated to **v7.5** (sec:quant, tab:ablations E-046
   row) and delivered. Full entry: EXPERIMENTS.md 14:30Z block.
4. **Operator approvals executed** ("Proposal sounds good, let's do it!"):
   (a) #468's box-local partial head WRITTEN OFF — gpu-box-32966687 is off
   the never-destroy list; (b) **warm-start quantization PROMOTED** and its
   affordance landed as **E-050 `--fsq-warmstart`** (6c83ed3: continuous→FSQ
   resume, four-key hole, rebased fit schedule, recipe-only workflow key, 7
   guard tests in `tests/test_fsq_warmstart.py`).
5. **New reporting rules** (today): `ml/CLAUDE.md` §0f — any cross-experiment
   status uses FOUR fixed sections (completed / new results + next steps /
   queued / proposed changes), and **every experiment entry opens with a
   plain-English TL;DR** in the completed AND queued sections.
6. **Portability**: the 13 session status figures + generator committed as
   `ml/figures/` (60a81ce, README registered in docs.html; docs suite
   11/11). Convention: a figure delivered in chat is committed there in the
   same session.

## In flight at ~15:45Z (ETAs are that snapshot — verify, don't trust)

- **#478 (E-045.1, K=144 — THE decisive span-vs-step rung)**: TL;DR — give
  the 5-day model the full 2 years of history; ~0.07–0.15 confirms context
  was the bottleneck, ~0.5 blames step size. 20k ≈22:00Z, drain
  ≈23:30–00:00Z, on gpu-box-48254133 (H100).
- **#476 (E-045.3, K=48 rung)**: the middle rung. Drain ≈21:30–22:00Z on
  gpu-box-40024079. VERIFY it pulled the published Z (no `embedding` records
  in ml-live-476) — at last check it had only logged `resumed`.
- **#472 (E-047b, 126.9M month-block codec)**: was in final probes at
  14:30Z — likely green; harvest + verify its checkpoint published, then
  queue-check/idle-burn its box (gpu-box-48383989).
- **#483 (E-047-HEAD 6th — fusion-vs-selection verdict, vs A2a's 0.0721)**:
  queued behind #478 on the same H100; must PULL `Z_8b639abe36_37e146384b`
  (the e047a codec's Z — its clean reappearance after the chimera deletion
  is CORRECT: the key is simply that codec's identity, republished fresh by
  #470's embed pass). Verdict ≈02:00–03:00Z.
- **#480 (E-049a d_z-6 control — the road-B session's)**: on
  gpu-box-32966687, ~200k ≈ tomorrow noon. ALSO the parent for the first
  **E-050 warm-start dispatch** — when it finishes, that dispatch is the
  promoted next move (recipe must set `fsq_warmstart`, `resume`,
  `fsq_levels`; contract in the E-050 commit and its tests).

## Scheduled wake-ups (bound to the writing session — a NEW session must
re-create what it needs; the hourly fleet-health check is account-level and
survives)

- 19:00Z: fleet verify + #472 harvest.
- 23:45Z: E-045 ladder verdicts → factorial figure
  (`ml/figures/fig_factorial2.py`) + paper sec:factorial + PDFs + the ladder
  summary to the operator (§0f format, TL;DRs, markdown links one per line).

## Still owed / deferred

Cross-framework Tier-1 audit (e047a-tpu vs #473 — RUNNABLE now, named next,
don't start unprompted). e048b arm B (held on the FSQ fork — E-046's win +
E-049's collapses argue the warm-start branch first). convert.py
args['data'] template provenance fix. TPU stage-2 for a follow-up K=144 head
(operator's call).

## Traps this session hit (beyond what ml/CLAUDE.md §7 already carries)

- A silent trainer death shows as: metrics frozen, GPU ~0%, live branch
  still force-pushing unchanged content. Diff steps-vs-time, not push time.
- `job_timeout` arithmetic at K=144: 1.212 s/step clean + **1,720 s per
  light probe × 10** ≈ 690 min before the ladder — 900 kills it, use 1400.
- `dispatch_run.mjs` requires a plan; `python3 ml/plan_schedule.py --steps
  20000 --lr 0.001 --schedule expdecay --cooldown-frac 0` reproduces the
  stage-2 head curve exactly (verified against #463's live LR).
- Concurrent sessions both edit EXPERIMENTS.md — `git pull --rebase` before
  writing, and re-read your own block after: a rebase clipped a heading of
  mine once today.
- The paper build dies silently on an unescaped `#` in .tex prose
  (`set -e` + batchmode); `bash ml/paper/build.sh` must end with its
  "built:" line naming BOTH PDFs, or it didn't finish.

# Session handoff · 2026-08-25 · Road B, the collapse, and the E-046 verdict

**What this is.** The complete handoff of the 2026-08-25 daytime session
(~10:00–16:25Z) that ran the road-B token programme, written INTO THE REPO so
any future session — including one under a different Claude subscription —
can resume from `git clone` alone. Read order for a cold start:
`ml/CLAUDE.md` (the rules, §0f included — this session added it), the top of
`ml/EXPERIMENTS.md` (e-049, e-046, the 08-25 triage blocks), then this file
for the operational state the log does not carry. A parallel session ran the
same day (E-050, the figures convention); its work is folded in below.

## 1 · The thread of the day, in one paragraph

Chris directed: continue **road B** — the paper-faithful FSQ token (one 2^16
token per pixel-bin, d_z 6, levels [8,8,8,5,5,5]) — *"very diligently,
tested with a decoder (a non-linear decoder)"*, and refuted the
per-step-error reading of the pentad −0.499 (the E-045 factorial's
context-span story is the accepted one). The session designed and dispatched
E-049 (plan, recipes, the new `--fsq-bound ln`, `ml/fsq_usage.py`
effective-bits instrument, the recon-audit adaptation); the cold-start FSQ
arm then **collapsed to a constant encoder** (#482 — three instruments
agreeing, the guard's #481 kill re-read as a TRUE positive); and the day
ended with the programme's best news anyway: **E-046 resolved at its
registered instrument** — lattice z out-forecasts continuous z — and the
approved repair, **E-050 warm-start quantization**, landed and gated on the
continuous parent #480.

## 2 · Ledger of the day's commits (both sessions)

| sha | what |
|---|---|
| `72a0921` | E-049: plan, recipes, `--fsq-bound ln`, `ml/fsq_usage.py`, workflow keys |
| `b0a3d7f`/`e17d0c2` | run numbers; #479 dead-GPU post-mortem, #481 re-dispatch |
| `fc5f62d` | #481 verdict (bound holds; guard retuned via `collapse_r` key), #482 dispatch |
| `e207acf` | **ml/CLAUDE.md §0f** (four-section status reports); #470 void + #483 queued |
| `5f40551` | **recon-audit adaptation** — float16 disease (3 modes, 12/40 channels survived), Argo-bin split scorer, uint16, npz path |
| `b0593d1` | #482 cancelled — constant-encoder collapse; #481 kill re-read TRUE positive |
| `6c83ed3` | *(parallel)* **E-050 `--fsq-warmstart`** — continuous→FSQ resume hole, 7 guard tests |
| `b3ee36a` | *(parallel)* Chris approves ~15:30Z; #468 partial written off |
| `bca7349` | **E-046 RESOLVED** — #477 ratio 0.4394; #480 exonerates d_z 6 |
| `60a81ce` | *(parallel)* `ml/figures/` convention + 13 status figures |
| `0a094f1` | `codec_io.html` + `token_roads.html` into ml/figures (Pages-served) |

## 3 · The findings, with their numbers

1. **E-046 RESOLVED: the FSQ codec earns its place at pentad.** #477 (seed 1,
   #427's exact 20k stage-2 window on run-455's lattice z): ratio
   **0.6497/1.4788 = 0.4394** vs A9 0.4916 and #427's 0.5036 (margin ~46× the
   pair spread). Seed-0 cutoff 0.4395 agrees. Caveats: lattice-z units; n=1
   registered on a new tier (§3b — the winner owes its pair). Compound
   reading: run-455's z is a measured ~32-bit sign code and STILL beats the
   1,024-bit continuous substrate — the capacity story's strongest evidence.
2. **Cold-start codec-side FSQ at d_z 6 collapses to a constant encoder.**
   #482 (guard off): persistence z-MSE 0.141 → 0.0009 → exactly 0; fit
   std_med 0.638 → 0.005; probe ZeroDivisionError — all by ~step 7.5k. The
   bound itself HELD (prequant_rms 1.0 at every fit, radii O(1) where
   unbounded runs sat at 3e4): a DIRECTION collapse under an intact scale.
   #480 (continuous d_z 6, same everything minus lattice+bound) is healthy —
   the natural experiment convicts the lattice-at-cold-start, not the width.
3. **The collapse guard's 0.05 threshold is uncalibrated on lattice z** (no
   FSQ codec was ever under it — #455 read NaN all run). New recipe-only key
   `collapse_r`; replacement monitor = the extended fit schedule's
   prequant_std_med/rms + `ml/fsq_usage.py` effective bits.
4. **recon_eval's float16 accumulation had THREE silent failure modes** at
   family-4 shape (26/40 channels inf-overflow; 2 saturate to
   exactly-constant series — undetectable by any finiteness guard; 1
   statistic corrupted). Fixed at `5f40551`; the Tier-1 audit + decoder-
   ceiling protocol (recon_decoder) is now runnable on pentad/d_z-6/FSQ
   codecs, with the per-(t,pixel) Argo-bin split the E-049 falsifier is
   registered on.
5. **Fleet lore paid twice**: #479 died on a box whose GPU was gone while its
   runner reported online+idle (a dispatch IS the health probe; Vast 48478310
   is a stopped lemon), and #470 went green-but-void (stage-2 OOM 13.2 GiB /
   24 GB — the d_z-64 block z needs the H100) while its embed published the
   block Z (`Z_8b639abe36`, embed-cache-v1), making the #483 re-run ~$2.

## 4 · Fleet and in-flight at 16:25Z

- **#478 (E-045.1, THE decisive span rung: 5-d steps, K=144 = 720-d span)** —
  H100 gpu-box-48254133, healthy; 20k ≈22:00Z. Pre-registered bands:
  ~0.07–0.15 confirms the context story at the hardest step size; ~0.5 the
  step story.
- **#483 (E-047-HEAD, 6th: fusion-vs-selection on the month-block z)** —
  queued behind #478 on the same H100; pulls the cached Z; verdict = 20k
  ratio vs E-045-A2a's 0.0721.
- **#480 (E-049a, continuous d_z-6 control AND the E-050 warm-start
  parent)** — gpu-box-32966687, healthy, lands ~05:00–12:00Z 08-26.
- **#476 (E-045.3, K=48 interpolating rung)** — 2nd H100 gpu-box-40024079.
- Finished today, comparisons owed: **#472/#473** (E-047b/a torch twins) —
  Tier-1 recon vs their TPU siblings, the cross-framework certificates.
- Boxes: Vast 48478310 (gpu-box-40623952) = dead-GPU lemon, STOPPED;
  48478309 (gpu-box-42005419) = STOPPED with **#482's ~15k corpse on disk**
  (the rescue step publishes it on that box's next job).

## 5 · Open work, in priority order

1. **E-050 first dispatch** — gated on #480 finishing. The AFFORDANCE is
   landed (`--fsq-warmstart`, recipe-only key, tests); **the recipe and plan
   doc do NOT exist yet** (`ls ml/recipes | grep warm` is empty). Needed: a
   `f4r2-40M-dz6-fsq65k-warm` recipe = the fsq65k recipe + `fsq_warmstart` +
   `resume !run-480…`, with §0d header and falsifier (same decoder-ceiling
   falsifier as E-049b; the cold-start collapse is the thing warm-start must
   not reproduce — watch fit std_med and effective bits from the first fits).
2. **Harvest #478 (~22:00Z), then #483, then #476** — each against its
   pre-registered reading; §0f-format the report.
3. **Diagnose #482's corpse** (cheap, CPU): where is the constancy born —
   pre-LN (encoder's doing) or post-LN only (the centering's)? Start box
   48478309, let the rescue publish `run-482`, run distinct-z counts +
   `ml/fsq_usage.py`. Informs whether E-050 keeps `ln` or switches to an
   RMS-only bound.
4. **E-046 seed-0 pair** (§3b: the winner buys its pair) — a seed-0 refinish
   must survive the gradient cliff that killed #468 at 15.8k.
5. **Tier-1 recon for #472/#473 vs the TPU codecs**; wire `fsq_usage.py`
   output into the audit JSON (currently run beside it).
6. **Paper v7.5** — the tex still says E-046's verdict is "dispatched and not
   in" and carries the pre-factorial cadence reading in places; E-045's
   resolution, E-046's verdict, the collapse finding and E-050 all belong in
   the next revision.

## 6 · Portability notes for a different subscription

- **Everything scientific is in this repo** — rules (CLAUDE.md, ml/CLAUDE.md),
  log, plans, recipes, figures (`ml/figures/` — the codec-I/O diagram and the
  Two Roads to a Token deck are Pages-served HTML there; the claude.ai
  artifact copies are superseded and account-bound).
- **Credentials are NOT in the repo** (deliberately, never commit them): the
  GitHub fine-grained PAT, the Vast.ai API key, CMEMS login, TPU/GCP and HF
  access notes live in the OLD claude.ai project's `claude/*-access.md` docs.
  A new subscription needs them pasted once into its own project (or a
  password manager). Without the PAT: clone and read works (public repo);
  push and dispatch do not.
- **Scheduled wake-ups are account-bound and die with the old session.** The
  old session holds a 22:51Z 08-25 evening-harvest trigger; a new environment
  must re-establish its own monitoring (the status page +
  `ml-live-<n>`/`ml-metrics` branches are the instruments; ml/CLAUDE.md §2).
- **Two sessions may write the log concurrently.** Working rule that held all
  day: `git pull --rebase` before every push, never force, never duplicate a
  harvest another session already recorded.

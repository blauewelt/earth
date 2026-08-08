# ML operations handbook — state of the program and how to run it

Written 2026-08-08 as the single onboarding document for a fresh session.
CLAUDE.md governs the app; this file governs the ML program. When they
conflict on ML matters, this file wins. Keep it current the way CLAUDE.md
is kept current: every hard-won fact lands here, not in a chat transcript.

Secrets are NOT in this repo. Locations: GitHub PAT `/home/claude/.gh_pat`
(sandbox; recreate the file from the claude.ai project doc
`claude/github-access.md` on a fresh container), CMEMS credentials in the
claude.ai project doc `claude/copernicus-marine-access.md` (env vars only,
never on disk, never in argv — the permission classifier blocks credentials
on command lines, correctly). The Vast.ai key: recreate
`/home/claude/.vast_key` from the claude.ai project doc
`claude/vast-access.md` on a fresh container (mirrored there 2026-08-08
after a parallel session's container lost it — the original chat session
still held it and wrote the doc). Auto-top-up is ON for the Vast account
(Chris, 2026-08-08): the $50 approved budget is the binding constraint,
not the credit balance.

## 1 · What this program is

Self-supervised "Earth-state embeddings" of the observed ocean, probed
against the AMOC. Three stages, strict gradient boundaries:

1. **Codec** (`ml/model.py`, `ml/train.py`): per-pixel-month masked
   autoencoder → z ∈ R^64. Transformer encoder over C channel tokens +
   context token + CLS; query-MLP decoder that never sees inputs.
   Self-supervised only. Sizes are configurable and stored in the
   checkpoint; `codec_from_ckpt()` is the ONE place that reconstructs
   architecture — never hand-construct `PixelMAE` from a checkpoint.
2. **Stage-2 temporal** (`ml/temporal.py`): causal transformer over frozen
   embeddings, predicts z_{t+1}. Also self-supervised. Embedding cache is
   codec-weight-hashed (`Z_<run>_<whash>.npy`) — the poisoned-cache bug
   (two withdrawn results) is why.
3. **Probes** (measurement, never training): `probe_kfold.py` (ridge; also
   `--probe mlp`), `probe_head.py` (attention head over the UNPOOLED
   section; `--raw` / `--raw --raw-patch` are the end-to-end controls;
   `--head-dim/--head-blocks` scale it). Year-blocked k-fold, block
   bootstrap. THE defensible number is the pooled-ridge k-fold; the head
   is the capability bound.

The paper (`ml/paper/`, PUBLIC in the repo since 2026-08-08, permalinks
mandatory with every delivery) is the canonical statement of results.
v4 is current. `ml/make_table.py --markdown` generates the master table
from the runs' own JSON — never transcribe numbers by hand.

## 2 · Results state (2026-08-08 afternoon)

- Probe-ladder finding: MLP ≈ ridge everywhere (nonlinearity adds nothing);
  attention head over unpooled section adds a lot. patch24 codec + head =
  0.690 [0.57, 0.78] (seed 1) / 0.654 [0.54, 0.75] (seed 2, run #43).
  SECOND SEED RETRACTS THE MARGIN (LEADERBOARD "The second seed"): the
  two-seed mean 0.672 vs raw-3x3 0.659 → +0.013 ≈ nothing; the earlier
  "+0.031, n.s." was one seed's noise read as a direction. The unpooling
  capability claim SURVIVES (both seeds far above every pooled probe).
  House rule since: no head claim on one seed. Pretraining's case rests
  on field prediction, MOVE transfer (0.206 [0.04, 0.35], CI excludes
  zero), and capacity scaling (10M + family-3 anchored runs, open).
- Channel families: 1st = 4 Argo levels (C=12–15), 2nd = 8 levels
  (C=24/25). NEVER cross-compare. Read C from the checkpoint, not run names.
- Steps: SETTLED by the 1M-step headroom run (#30, harvested as
  patch24_1M): every probe metric flat from the 50k probe to step 1M
  (linear r 0.43–0.48, chan% pinned at 32); final k-fold 0.571 — within
  seed noise of 40k. Pilot size is CAPACITY-limited, never train it
  longer. 31,248 steps = 1 epoch at batch 512 on the global tensor.
- Head capacity sweep: BOTH columns degrade at 325k params (raw
  0.659→0.590, emb 0.690→0.611); the ~47k head is already past the
  optimum for 220 labels/fold, margin persists at both sizes. 2.0M
  cells running locally as a formality.
- Chinchilla policy (user-mandated): after every data change, recompute
  observed-values/epoch ÷ 20 → codec size anchor, record it here and in
  SCALING.md. Current C=24 global: ~270M values → ~13M params → the 10M
  codecs. Family-3 NA 0.25°: **822.6M MEASURED → 41.1M anchor** (§5;
  built 2026-08-08). Global 0.25°: ~5B+ → ~250M (estimate).
- Withdrawn: global14b + global15sst stage-2 numbers (poisoned cache).
  SAMBA is unprobeable (anti-generalizes even for raw wind). OSNAP shows a
  possible MLP-only positive — unreplicated hypothesis.

## 3 · Infrastructure

- **Fleet**: three Vast.ai RTX 4090 self-hosted runners (~$0.24/h each,
  ~124 steps/s at pilot size). Manage: `node scripts/gpu_box.mjs
  offers|create|list|stop|start|destroy` (offers ranks by TRUE cost incl.
  storage; DISK=50 — storage is billed while STOPPED and one host charged
  $0.167/h for it). Stopped box keeps its runner registration. The PAT
  mints registration tokens itself (Administration:write). Budget: $50
  approved by Chris, ~$5–6 spent (estimate as of 13:45 UTC 2026-08-08;
  exact figure needs the Vast key back — see the header).
- **Dispatch**: `node scripts/fleet_dispatch_wf.mjs '<inputs-json>' main`.
  Watch: `fleet_list_runs.mjs`, `fleet_run_state.mjs <id>`,
  `fleet_step_log.mjs <id> <regex>` (logs only after job end),
  `fleet_box_detail.mjs` (boxes + runners + costs).
- **Workflow** (`.github/workflows/ml-train.yml`): inputs cover steps,
  batch, d_z, patch, codec size (codec_d_model/layers/heads/d_dec),
  lr_floor/lr_decay_steps (decay-then-constant headroom protocol),
  max_minutes (re-fits cosine to the wall clock; hosted jobs DIE at 360
  min with no checkpoint otherwise), runner (self-hosted|ubuntu-latest),
  job_timeout. Runs on the box seed `/opt/earth-cache` from the
  **data-cache-v1 release** (13.8 GB, 8 assets) — COMPLETENESS-checked,
  because partial caches from failed fetches must not suppress seeding.
  Origin archives are flaky (SIO 504s killed five runs in one day); no
  cold box should ever contact them again. Publisher:
  `scripts/data_release.mjs` (streams tar|split, 1 chunk on disk).
- **Checkpoint mirror**: release `model-checkpoints-v1` holds every
  harvested codec as `<run>__pixelmae.pt` (~4 MB each at pilot size).
  This is the DURABLE copy — Actions artifacts expire in ~90 days and
  the sandbox is ephemeral (it has already eaten the paper once).
  STANDING RULE: after every `harvest_run.mjs`, upload the new
  checkpoint to this release. Restore = download + `codec_from_ckpt()`.
- **Sandbox limits**: ~7 GB RAM. The recurring OOM class: expressions that
  read like cheap slices but materialize the whole tensor —
  `clim[moy][..., c]` (use `clim[moy, :, :, c]`), re-indexing `d["X"]`
  (decompresses the whole array on every access — hoist it), `d["X"].copy()` (npz
  already returns fresh), out-of-place `nan_to_num` (use copy=False after
  taking the mask). Z built via memmap into the cache file. ONE heavy job
  at a time locally.
- **Push**: git proxy blocks; use `node scripts/git_api_push.mjs
  --token-file /home/claude/.gh_pat --branch main`. Never pipe through
  tail in && chains. After push, local shows "ahead" but `git diff HEAD
  origin/main` is empty (API mints new SHAs) — verify then reset --hard.
- **Paper**: rebuild BOTH pdfs on every substantive edit — `cd ml/paper
  && bash build.sh` (add `--figs` after data changes) — commit, push
  FIRST, deliver via SendUserFile WITH permalinks. build.sh generates
  paper_dark.tex (never hand-edit it).
  Project docs `paper/paper.tex`/`paper/make_figs.py` are the backup.
- **Status page**: https://blauewelt.github.io/earth/status.html —
  single-file phone dashboard (`status.html` at repo root) showing
  ml-train runs, live loss curves from `ml-live-<run>` branches, and
  release stats; reads ONLY public unauthenticated endpoints (2
  api.github.com calls per 2-min refresh + raw.githubusercontent.com,
  which is uncounted), so it holds no credentials.

## 4 · In flight right now (updated 2026-08-08 ~13:45 UTC)

- #40 DONE 14:10: harvested as **patch24_10M_60k** (10.26M params) +
  mirrored. Ridge 0.578 [0.451, 0.695] — first 24-channel codec back
  above the wind-only line; MOVE 0.238 (CI excludes zero, lowpass
  0.677). LEADERBOARD has the writeup. #44: FAMILY-3 pilot control
  (0.92M, 40k) — build_family3.py COMPLETED on-box, training since
  ~13:55 at ~5 steps/s (gather-bound at 0.25°: patch-3 gathers from a
  10.9 GB CPU tensor; ~2.2 h for 40k. Optimization if it matters:
  move Xt/OBS to the 4090's 24 GB and gather on-device). #46: 10M 30k
  re-re-dispatch (Brazil box, fixed seed code, in seed ~40 min — its
  link is slow; the per-chunk cap is 30 min so it either lands or dies
  loudly). #47: family-3 anchored 41M (576/10/8/768, 60k) — running on
  the box #40 freed. When done: harvest (f3_pilot_40k / f3_anchor41M /
  patch24_10M_30k — read C and patch from the checkpoint before
  naming), mirror checkpoints, probe_head on the 10M codecs with TWO
  seeds before any margin claim (house rule above).
- #42 (the first 10M 30k re-dispatch) HUNG 90 min in the seed step: the
  streamed `curl | tar` had no --max-time and a stalled connection on
  the Brazil box hung the pipe forever (release download counts showed
  chunks ac/ad/ae never requested — that is how you diagnose it from
  outside; logs are unreadable until a job ends). Cancelled; every seed
  curl now carries --max-time, deliberately NO --retry inside the tar
  stream (a mid-body retry restarts at byte 0 and corrupts the tar).
- #43 patch24_40k seed2: SUCCESS, harvested to ml/runs/patch24_40k_seed2
  + rekeyed (per-run JSONs say run="actions"; make_table needs the key =
  dir name) + checkpoint mirrored. Ridge 0.531 = exactly the wind-only
  line; head 0.654 → the margin retraction in §2 / LEADERBOARD / paper.
- TWO CLAUDE SESSIONS worked this program in parallel on 2026-08-08
  (~12:00–13:45): the interactive noon session (harvested #43's head
  probes, pushed the retraction + paper v4) and the scheduled family-3
  builder session (this handbook update). It worked — git rebase merged
  cleanly, the checkpoint upload collided harmlessly ("already_exists")
  — but check `git log origin/main` and the release assets BEFORE
  redoing any harvest/upload/doc edit: the other session may have done
  it already. The handbook is the coordination medium; update it first.
- pixel24_40k_seed2 checkpoint (from failed #39; patch=1 control) —
  probe suite run by the noon session (head 0.564; LEADERBOARD).
- Boxes: 3 runners online (45318655, 47094145, 35586926-Brazil). All
  busy or about to be; nothing to park. Spend: Vast key lost (header),
  so unbilled-API estimate only: ~$4 at 12:15 + ~1.5 h × ~$0.75/h ≈
  **$5–6 of the $50 cap**. The Brazil box's GitHub link is slow/flaky —
  it is the one that hung #42 and it cold-starts every seed; if it
  misbehaves again, destroy and re-provision US-side (gpu_box.mjs offers
  ranks true cost — needs the Vast key back first).

## 5 · Data ladder (user-approved: data first, then training)

- (a) DONE behind flag: `build_dataset.py --start-year 1982` (pre-GLORYS
  months = missing tokens; FC 1982–92 becomes truth).
- (c) DONE behind flag: `EARTH_RG_LEVELS=16` in fetch_rg_channels.py.
- (b) IN PROGRESS: `ml/fetch_cmems025.py` fetched + assembled
  `ml/cache/base025_na.npz` — 0.25°, NA window, 1993-01..2024-12, T=384,
  84,405 ocean px (14.6x the 1° window), channels cur_speed/log_mld/**ssh**
  (zos served!). Sanity-checked (Gulf Stream 0.70 m/s vs gyre 0.08).
  MIRRORED as release asset `base025_na.npz` on data-cache-v1 — the
  family-3 builder session does NOT need CMEMS credentials to get it.
  2025–26 need the interim 1/12° product binned to 0.25 (same pattern as
  refresh_data.py glorys tail).
- (d) queued: GRACE ocean bottom pressure, ERA5 (SLP/fluxes/10m wind),
  satellite SSS, sea ice, ocean color — from the AlphaEarth gap analysis
  (their modality list translated to ocean; they are terrestrial-only).
  SSH already arrived via (b). OBP/ERA5 need Earthdata/CDS accounts.

### Family-3 builder — DONE 2026-08-08 (was "the next task")

`ml/build_family3.py` → `ml/cache/family3_na025.npz`, built and
validated in-sandbox (memmapped X + per-month slab passes; the dense
tensor is 10.9 GB against 7 GB RAM), then wired into the workflow as
the `tensor` input (`family2` default | `family3_na025`) with `--data`
plumbed through train.py and every probe. As-built facts:

- [T=516 (1982-01..2024-12), H=281, W=481, C=39] · 84,405 ocean cells ·
  chan = cur_speed/log_mld/ssh + rg_t/rg_s at 16 levels + tau_x/tau_y/
  tau_x_std/tau_y_std. truth_fc 490 in-axis months (the 1982-92 cable
  decade is in). npz is 3.0 GB compressed; `recipe` key = skip guard on
  boxes (bump RECIPE_REV on any recipe change).
- **Chinchilla, measured at build: 822,649,104 observed values (base
  97.2M / rg 551.2M / wind 174.2M) → anchor 41.1M params.** Above the
  600–800M estimate (RG covers 41.4% of ocean×months at 0.25°). Train
  pool 30.38M pixel-months → 1 epoch at batch 512 = 59.3k steps.
  Recorded in SCALING.md (with the 1°-smoothness caveat: 41M is a
  ceiling, not a target).
- Wind mean AND std both come from the NCEP DAILY files now (one
  source); 1982-92 daily files are fetched from PSL on first build and
  cached — they are NOT in the wind_daily release tar (1993+ only).
  Open: extend the release with them so cold boxes stay offline.
- RG extension months enumerate from the LOCAL cache, never the SIO
  index scrape. base025_na.npz seeds from the release in the workflow
  (and build_family3.py self-fetches it if absent — no CMEMS needed).
- Dispatched: #44 pilot control (0.92M, 40k, family-2 champion
  protocol), #47 anchored 41M (576/10/8/768 = 40.7M, 60k ≈ 1 epoch,
  temporal_steps=0 to spare box disk). LEADERBOARD has the family-3
  block; paper §families waits for the first results (no numbers yet).
- Still open for family 3: 2025–26 base tail (interim 1/12° binned to
  0.25, needs CMEMS), wind-only ridge baseline ON THIS TENSOR (family-2
  0.531 does not carry over), steps-matched anchored 40k control if the
  60k result warrants it.

## 6 · Standing user directives (verbatim intent)

- Data work first, then training runs (2026-08-08).
- Chinchilla arithmetic redone at every data change; scale codec to match.
- Paper: rebuild + deliver on every update, always with permalinks.
- No AskUserQuestion dialogs; choices in prose (CLAUDE.md §0).
- Budget cap $50 for boxes; park idle boxes ($0.014/h stopped).
- Zenodo/DOI archive planned when results settle; move the data mirror
  there when it outgrows GitHub (~30 GB threshold) or at submission.

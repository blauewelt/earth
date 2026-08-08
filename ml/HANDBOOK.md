# ML operations handbook — state of the program and how to run it

Written 2026-08-08 as the single onboarding document for a fresh session.
CLAUDE.md governs the app; this file governs the ML program. When they
conflict on ML matters, this file wins. Keep it current the way CLAUDE.md
is kept current: every hard-won fact lands here, not in a chat transcript.

Secrets are NOT in this repo. Locations: GitHub PAT `/home/claude/.gh_pat`
(sandbox), Vast.ai key `/home/claude/.vast_key` (sandbox), CMEMS
credentials in the claude.ai project doc `claude/copernicus-marine-access.md`
(env vars only, never on disk, never in argv — the permission classifier
blocks credentials on command lines, correctly).

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
  **0.690** [0.57, 0.78]. Attribution matrix: raw-1px 0.613 / emb-1px 0.617 /
  raw-3x3 0.659 / emb-3x3 0.690 → pretraining margin **+0.031, n.s.**
  Pretraining's case rests on field prediction, MOVE transfer
  (0.206 [0.04, 0.35], CI excludes zero), and capacity scaling (open).
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
  codecs. Family-3 NA-window: ~600–800M → 30–40M. Global 0.25°: ~5B+ →
  ~250M.
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
  approved by Chris, ~$4 spent.
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

## 4 · In flight right now (updated 2026-08-08 ~12:15 UTC)

- #36: 10M codec 30k (NL box, in Train). #40: 10M codec 60k re-dispatch.
  #41: patch24_40k seed2 re-dispatch (queued). When done: harvest, run
  probe_head on the 10M codecs (capacity-vs-margin, THE open question),
  and patch24 seed2's head vs 0.690.
- pixel24_40k_seed2 checkpoint HARVESTED from failed #39 (train+upload
  succeeded before the disk death) — probe suite still to run locally.
- patch24_1M harvested + rekeyed (per-run probe JSONs said run="actions";
  make_table needs the key = dir name) + in LEADERBOARD/paper/table.
- Local: head capacity sweep 2.0M cells (raw-L then emb-L) running.
- Box churn 2026-08-08: gpu-box-46045353 (Virginia) hit 50/50 GB disk,
  runner died mid-#39 → destroyed, replaced by instance 47171781
  (gpu-box-35586926, 129GB RAM Brazil, $0.267/h) — its cold start is the
  first real test of release seeding (watch download counts rise).
  Workflow now prunes disk below 8 GB free and seeds truth/ from the
  release (OSNAP server timeouts killed #37/#38 at build).
- A scheduled fleet-check fires 12:43 UTC (harvest, retries, box parking).

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

### THE NEXT TASK — the family-3 builder (spec)

Goal: `ml/build_family3.py` producing `ml/cache/family3_na025.npz`:
1. Base: `base025_na.npz` (cur_speed, log_mld, ssh at 0.25°).
2. Extend axis to 1982-01 (missing tokens pre-1993), reusing the
   --start-year pattern from build_dataset.py.
3. RG at `EARTH_RG_LEVELS=16` (fetch_rg_channels logic, refactored to
   write onto the 0.25° grid by bilinear upsample from 1°; keep a note
   that RG is intrinsically 1°-smooth).
4. NCEP wind stress + within-month std onto 0.25° (bilinear from the
   gaussian grid; fetch_wind_channels/fetch_wind_stats logic).
5. Truth series unchanged (fetch_truth.py attaches to the axis).
6. Anomaly/holdout machinery must NOT assume grid: check
   anomaly_transform and probes for 1°-hardcoded windows — the holdout
   lon block (-45,-25) and RAPID section clip work in degrees, fine, but
   memory at 0.25° NA ≈ 384x281x481x~24ch x4B ≈ 5 GB dense → the SANDBOX
   CANNOT hold anomaly copies; build and train on the BOXES (64 GB), or
   build here with memmaps. Expected total ~600–800M observed values →
   Chinchilla anchor 30–40M params (record the real number in SCALING.md).
7. Then dispatch family-3 codecs (pilot-size control + anchored-size) via
   the workflow; a `tensor` input for the workflow build step (which
   tensor recipe to run) will need adding — currently it always builds the
   family-2 recipe.
8. New family block in LEADERBOARD + paper §families; never cross-compare.

## 6 · Standing user directives (verbatim intent)

- Data work first, then training runs (2026-08-08).
- Chinchilla arithmetic redone at every data change; scale codec to match.
- Paper: rebuild + deliver on every update, always with permalinks.
- No AskUserQuestion dialogs; choices in prose (CLAUDE.md §0).
- Budget cap $50 for boxes; park idle boxes ($0.014/h stopped).
- Zenodo/DOI archive planned when results settle; move the data mirror
  there when it outgrows GitHub (~30 GB threshold) or at submission.

# E-069 handover — the cone-native codec, end to end, for whoever picks it up next

**Written 2 Sep 2026, against main at `ac41b97`.** Companion to
[`ml/plans/E069_cone_codec.md`](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E069_cone_codec.md)
(the design and the hypotheses) and
[`ml/figures/geofm_survey/GENERIC_EMBEDDING_INPUTS.md`](https://blauewelt.github.io/earth/docs.html?f=ml/figures/geofm_survey/GENERIC_EMBEDDING_INPUTS.md)
(the argument). This file is the *procedure*: every step in the order it has to
happen, what each step produces, and how to check that it did.

> **Who this is for.** An agent or person who has NOT read the deck, has NOT
> designed the codec, and is being asked to (a) get the r3 tensor built and
> published, (b) run the codec on a GPU box or a TPU, (c) run the H1 velocity
> probe and read it, (d) wire the stage-2 arms, and (e) port the codec to
> JAX so it can run on the TPU pool. Each phase below is written so that it
> can be executed without the others being understood, and each ends with
> the artefact that proves it happened. Where a step has a known trap, the
> trap is stated before the step, not after.
>
> **Honest status of the plan it accompanies.** `E069_cone_codec.md` is a
> design specification with a status line. Steps 1–4 of its §6 are done and
> tested; steps 5–8 are one paragraph each. This document expands those
> paragraphs into procedures. Nothing in it has been run on the ocean tensor
> yet; every number quoted as "expected" is either a CPU-smoke measurement
> (§3) or a pre-registered prediction (§1), and is labelled as such.

**Standing rules that bind every phase** (from `ml/CLAUDE.md`; the section
numbers are that file's):

- Verify the ARTEFACT, not the intention (§0.1). A step is done when the
  file it produces exists, has the size and hash it should, and a reader can
  open it. "The script printed success" is not that.
- A run number never travels alone (§0c): `#NNN (E-069 codec, seed 0)`.
- Every dispatch opens with the structured header (§0d) — the one in
  `E069_cone_codec.md` §0 is the template.
- Hypothesis and falsifier are written into `ml/EXPERIMENTS.md` BEFORE the
  dispatch (§1). They already are (E-069's entry); a dispatch adds its run
  number and its control's run number to that entry, nothing else.
- The session plans; Opus subagents implement (§0b); the session runs the
  tests on what a subagent wrote before committing it.
- Update `claude/expectations.md` (project doc) and `ml/OVERVIEW.md` (with
  its stamp) in the same breath as dispatching or harvesting (§5.27, §0g).
- Links in chat are markdown links, one per line (root `CLAUDE.md` §0b).

---

## 1 · The experiment in one page

**What it changes.** Today's codec (`PixelMAE`) encodes one 3 × 3 patch of
one pentad per channel. Nothing that needs two snapshots — a displacement, a
tendency — can be in its embedding, so stage 2 rebuilds motion from compressed
codes. E-069's codec (`ConeMAE`) encodes the *inner dependency cone*: the
same lag-0 patch plus, for every channel, a sunflower of "dots" at lags
1…6 pentads whose reach follows the channel's physics (fast/wide for wind,
slow/narrow for currents). Stage 2 keeps the *outer cone* — lags 7…143 with a
reach that grows with lag — over the embeddings. The union of the two
stencils is the whole dependency cone; the overlap is the anchor column only.

**The three hypotheses, each with the reading that kills it** (plan §1):

| id | claim | measured by | falsified if |
|---|---|---|---|
| H1 | the cone embedding carries local velocity | ridge R² from frozen z to `cur_u`, `cur_v` at the anchor with the `cur_*` inputs hidden (`velocity_probe.json`) | cone and snapshot codecs agree within the seed interval, n ≥ 3 each |
| H2 | that buys rolled skill at leads 1–6 (5–30 d) | the #516 battery: per-lead `acc` and `msss_clim` vs climatology / persistence / damped persistence / LIM, block-bootstrap intervals, trained and held-out longitudes separately | leads 1–6 agree with the E-064b control within the five-seed interval |
| H3 | the split is not zero-sum | the same head on the annulus-only stencil vs the full cylinder | the annulus head is worse by more than the interval |

**Pre-registered predictions** (plan §1, written before any run): H1 holds
with R² 0.3–0.6 against < 0.1; H2 holds at leads 1–3 for `cur_*` and `ssh`,
not for `sst`; H3 holds. What decides the programme's next step is H2.

**Order of spending.** H1 first — it is one codec training (~3 h) plus a
ridge, and its answer decides whether the 15 stage-2 heads of H2/H3 are
worth ~30 box-hours. If H1 is falsified, stop after it and write that up; a
falsified H1 with an H2 that somehow held would be a result nobody could
explain.

**The velocity choice, so nobody re-litigates it.** L_in = 6 pentads (30 d)
and v_B = 0.3 m/s. One 0.25° cell per pentad is 0.06 m/s; eddies (0.1–0.3
m/s) resolve at lag 1, Rossby waves (0.03 m/s) resolve inside 6 lags,
atmospheric forcing decorrelates in 1–2 pentads, SST anomaly memory (3–6
months) is longer than the window and is left to stage 2 on purpose. Plan §2
has the paragraph; do not shorten the window without re-deriving it.

---

## 2 · Inventory — every file, what it exposes, what to call

All paths are repo-relative. All of this is on `main` and tested
(42 tests: `tests/test_cone_geometry.py` 27, `tests/test_cone_smoke.py` 7,
`tests/test_e069_family4_r3.py` 8).

### 2.1 `ml/cone.py` — the geometry (numpy only, no torch)

| symbol | signature | what it returns |
|---|---|---|
| `FAMILIES` | dict | `A`/`B`/`C`/`rg` families: speed `v` (m/s), memory `tau_days`, `L_corr_km`, `lags` |
| `channel_family(name)` | `str → "A"|"B"|"C"|"rg"` | RAISES on an unknown channel name — a channel with a guessed family is a channel whose cone is a guess |
| `is_depth_channel(name)`, `channel_depth_dbar(name)` | | `rg_*` channels carry their pressure level; surface channels return 0 |
| `reach_km(family, lag_pentads, dt_days=5.0)` | `→ float` | B: `max(100, 0.3·86400·5·(1+ℓ)/1000)` = 129.6, 259.2, …, 907.2 km for ℓ = 0…6; A: 500 km at ℓ ≤ 1 else 0; C: `max(r_B(ℓ), 500 at ℓ ≤ 1)` |
| `slots(r_km)` | `→ int` | `clamp(round(24·(r/900)²), 6, 24)` dots on a ring |
| `inner_dots(lat_deg, family, L_in=6, dlat_deg=0.25)` | `→ dict of arrays (lag, dy, dx, dy_km, dx_km, …)` | the inner cone for one family at one latitude: anchor column per lag + a Vogel sunflower (`temporal.spiral_offsets`, r_min 28 km, ramp 0.5, aspect 0.71). Family A: lag 1 only; `rg`: anchor column only |
| `channel_dots(lat_deg, name, …)` | | `inner_dots` for the channel's family |
| `outer_reach_km(k)` | `→ float` | `min(4444, max(111, 0.3·86400·5·(1+k)/1000))` |
| `outer_spiral(lat_deg, k, dlat_deg=0.25, n_pts=24, L_in=6)` | `→ (dy, dx) int arrays` | the stage-2 ring at lag k between `r_lo = r_in(k)` (k ≤ 6) and `r_hi = outer_reach_km(k)`; EMPTY for k ≤ 6 by construction; first ring at k = 7 |
| `ground_km(dy, dx, lat_deg, dlat_deg)` | | signed ground offsets, cos φ applied zonally |
| `coverage_report(lat_deg, L_in=6, K=144, …)` | `→ dict` | asserts union = cone and overlap = anchor column on the grid; the 3,432-vs-20,880 cell count |
| `budget(lat_deg, chan_names, L_in=6, …)` | `→ dict` | token count per anchor: 42 patch + 706 dots = **748** on the r3 list |

Pinned numbers (`tests/test_cone_geometry.py`): inner dot counts per channel
A 8, B 80, C 81, rg 6; budget 748 at 30° N and latitude-independent.

### 2.2 `ml/cone_sampler.py` — gathering a batch (numpy only)

`ConeSampler(X, OBS, lats, lons, chan_names, L_in=6, dlat_deg=0.25, future_lags=(1, 2))`

- `X`, `OBS` are `[T, H, W, C]` and are indexed lazily (memmap-safe; never
  materialised). `OBS` may be `train_cone.FiniteView(X)` for a tensor with no
  mask.
- `.sample(anchors[B, 3] of (t, y, x)) → dict`: `vals, obs, valid, chan,
  dy_km, dx_km, lag_days, depth` all `[B, N]` (N = max dot count in the
  batch, padding is `valid=False`); `patch_vals, patch_obs [B, C, 9]`;
  `fut_vals, fut_obs [B, C, F]`; `ctx [B, 4]` = sin/cos season, lat/90,
  lon/180; `anchors` echoed. Off-grid dots are invalid (no longitude wrap).
- `.n_dots(y)`, `.row(y)`: the per-latitude dot table (cached).
- `.bin_span() → (lo, hi)`: the lags a cone reads (−L_in … +max(future)).
- `.admissible(anchors, train_bins[T] bool) → bool[B]`: every bin the cone
  reads is a training bin (depends on t only).
- `.certify(anchors, train_bins) → int`: brute-force count of anchors whose
  cone touches a non-training bin. The trainer REFUSES if this is non-zero.

Measured in the sandbox: 256 anchors gather in 0.1–0.2 s from a 4.5 GB
memmap. **The gather is host-bound; the network is not the bottleneck.**

### 2.3 `ml/cone_codec.py` — the model (torch)

`ConeMAE(n_chan, d_model=256, n_heads=8, n_latents=64, n_layers=6, d_z=32, d_dec=256, dec_layers=2, n_fourier=8)`

- `tokens(b) → (toks [B, 2+C+N, d_model], kpm [B, 2+C+N] bool)`; order is
  `[cls, ctx, patch × C, dots × N]`; `kpm` True = ignore. **Existence is
  enforced here and nowhere else** (module docstring) — an invalid dot's
  token is still built and its contents can never reach z; the smoke test
  perturbs one and asserts `torch.equal` on z.
- `encode(b) → (z [B, d_z], latents [B, n_latents, d_model])`.
- `query_tokens(chan, dy_km, dx_km, lag_days, depth) → [B, Q, d_dec]`;
  negative `lag_days` is a future query.
- `decode_from_z(z, q) → (mu, logvar)` — the HEADLINE path (z alone).
- `decode(z, latents, q)` — the auxiliary path; weight `aux_latent_w`
  (default 0.25, `0` = the z-only codec exactly). See the docstring for why
  the headline loss cannot go through this path.
- `forward(b, plan) → {"loss", "z", "terms"}`; `terms` carries `nll`, `mse`,
  `wsum`, `n_targets`, `logvar_mean`, `frac_chan_masked`, `frac_dot_masked`,
  and `nll_latent` when the aux term is on.
- `default_plan(chan_names, cur_drop=0.5, other_drop=0.3, lag_band_p=0.3, sector_p=0.3, future=True, anchor_recon=True, n_dot_queries=256, aux_latent_w=0.25, future_lags=(1, 2), generator=None, device=None)`.
- `FAMILY_W = {"A": 1.0, "B": 2.0, "C": 1.5}`; `LOGVAR_MIN/MAX = ±8`;
  `nll_gauss`, `signed_log`, `family_weights`.
- `param_count()`: **7,048,994 at 42 channels** with the defaults; 253,538
  at the smoke geometry.

The batch `b` is `train_cone.to_torch(sampler.sample(anchors), chan_depth,
device)`: the sampler's arrays as tensors plus `chan_depth [C]` (per-channel
dbar from `cone.channel_depth_dbar`).

### 2.4 `ml/train_cone.py` — the trainer (torch)

```
python3 ml/train_cone.py --tensor <path.npz> [--holdout-scope window]
    [--holdout-years 2009,2017,2023] [--steps 20000] [--batch 256] [--lr 3e-4]
    [--seed 0] [--d-model 256] [--n-heads 8] [--n-latents 64] [--n-layers 6]
    [--d-z 32] [--d-dec 256] [--dec-layers 2] [--n-fourier 8] [--L-in 6]
    [--future-lags 1,2] [--n-dot-queries 256] [--aux-latent-w 0.25]
    [--eval-every 0] [--eval-anchors 1024] [--save-every 0] [--certify-n 4096]
    [--velocity-probe] [--snapshot-ablation] [--probe-anchors 2048]
    [--out ml/runs/cone] [--metrics metrics.jsonl] [--smoke]
```

What it does, in order: `load_data` (tensor → anomaly space via
`trainprobe.anomaly_transform`, the ONE transform; a read-only memmap gets a
writable scratch copy `<tensor>_cone_scratch.npy` — **~36 GB on the pentad
tensor, budget the disk**) → `admissible_bins` → **pool certificate** (refuses
on any violation) → `ConeMAE` → AdamW (wd 0.01) + cosine to zero + grad-clip
1.0 → held-out eval at step 0 and every `--eval-every` on a FIXED anchor set
with a FIXED mask seed → checkpoint `cone_codec.pt` every `--save-every`
(default steps/4) and at the end → optional velocity probe → optional
snapshot ablation (`--L-in 0` twin, `metrics_snapshot.jsonl`,
`snapshot_codec.pt`) → `velocity_probe.json`.

Outputs in `--out`:

| file | content |
|---|---|
| `metrics.jsonl` | `ml/train.py`'s record family: `{"config": {…}}` first (with the additive cone fields `trainer, arm, L_in, n_latents, n_dot_tokens, future_lags, aux_latent_w, holdout_scope, holdout_years, lr, seed`), then `{"step", "loss_rec", "loss_nei"}` (= train nll, train mse) every steps/200, and `{"step", "held_out_nll", "held_out_mse", "held_out_targets", "wall_s"}` at every eval. **status.html already renders this family; no page change is needed.** |
| `cone_codec.pt` | `{"args", "model", "chan_names", "norm", "step", "arm", "L_in", "params"}` — `norm` carries `space="anomaly"`, `dynamic`, `holdout_years`, `tensor_norm` |
| `velocity_probe.json` | `{"cone": {"cur_u": {"r2", "r", "n", "probe"}, "cur_v": …, "folds", "n_anchors", "d_z"}, "snapshot": …, "delta_cur_u", "delta_cur_v", "L_in", "steps", "seed"}` |

The probe uses `probe_kfold.kfold_r` (year-blocked folds, the programme's
ridge) and falls back to a local ridge only if that import fails; the JSON
says which. Probe anchors are sorted by bin before the ridge — the reason is
in the code comment (the inner-tail λ selection needs time order).

### 2.5 `ml/build_family4.py --rev r3` — the data

`--rev r3` writes `family4_na025_pentad_r3.npz` (recipe string `f4r3`):
r2's 40 channels bit-identical, plus `cur_u` (index 40) and `cur_v` (index
41), the binned mean GLORYS12 components `cur_speed` is the hypotenuse of.
No new download. `CHANS_BY_REV["r3"]` is the channel list. An r2 cache is
refused for an r3 build.

### 2.6 `ml/recipes/f4r3-cone-5M.json`

Selects the r3 tensor and f4r2-40M's codec settings. Its `_description`
says plainly that the cone flags are NOT in it because the workflow has no
`$RECIPE_<KEY>` for them yet (§5 below adds them).

### 2.7 Tests

```
python3 -m pytest -q tests/test_cone_geometry.py tests/test_cone_smoke.py tests/test_e069_family4_r3.py
```
42 passed, ~2 min on CPU (torch CPU wheel required:
`pip install --break-system-packages --index-url https://download.pytorch.org/whl/cpu torch`).

---

## 3 · Run it locally in ten minutes (do this first, whatever phase you own)

```
cd <repo>
python3 -m pytest -q tests/test_cone_geometry.py tests/test_cone_smoke.py tests/test_e069_family4_r3.py
python3 ml/train_cone.py --smoke --steps 200 --batch 32 --eval-every 50 \
    --velocity-probe --snapshot-ablation --out /tmp/cone_smoke
```

What the smoke MUST print (measured 2 Sep 2026, CPU, 39 s for the cone arm):

- `X [T=120 H=40 W=56 C=8]` and the channel list
  `cur_speed, log_mld, ssh, tau_x, tau_y, sst, cur_u, cur_v`.
- `[cone] L_in=6 · <N> dot tokens + 8 patch tokens per anchor`.
- `[cone] pool certificate: 0 violations in 4096 drawn anchors` — anything
  else is a bug in the sampler or the pool, not a warning.
- `[cone] ConeMAE 253,538 params (0.254M)`.
- `held-out nll` falling: 2.037 at step 0 → ~1.75 at step 200 (seed 0).
- `[probe] cone cur_u R2 +0.07…` against `[probe] snapshot cur_u R2 −0.01…`
  (seed 0: +0.073 vs −0.015; same sign at seeds 1, 2). The planted shear
  flow has R²(velocity | one frame) = 0.0002 and R²(| two frames) = 0.906,
  so the sign is the mechanism and the size is a toy's.
- `/tmp/cone_smoke/velocity_probe.json` exists and `"probe"` reads
  `"probe_kfold.kfold_r"` (if it reads `"local ridge"`, the import failed —
  fix the environment before believing any number).

---

## 4 · Phase A — the r3 tensor: build once, pin, publish, stage

**Trap first.** `np.savez` stamps zip timestamps, so a rebuilt family-4
tensor never hashes the same twice even from identical inputs, and the
embed cache is keyed on that hash (`ml/CLAUDE.md` §7). The r3 tensor is
therefore built ONCE, its sha256 becomes the pin, and every later consumer
PULLS it. Two boxes that each build their own are two experiments.

**Inputs the builder needs** (all public, no credential): `base025_na.npz`
and the RG-Argo tars and the wind tars from the `data-cache-v1` release;
`truth_pentad.npz`; `pentad025/{index.npz, pentad_mean_{uo,vo,mlotst,zos}.npy}`
and `sst_na025/{index.npz, sst_daily_na.npy}` from the Hub
(`https://huggingface.co/datasets/chfrank/earth-tensors`). The workflow's
family-4 branch (`.github/workflows/ml-train.yml` ~`:791–887`) fetches all
of them; `seed_sst` (~`:725–747`) is FATAL on failure by design — a missing
SST would otherwise train silently as an all-missing channel.

**Route A1 (recommended): build on a Vast box through the workflow**, after
the §5.1 wiring lands (the tensor stem only; §5.2 is not needed for this).
The publish mechanism already exists: `window: publishtensor` makes the
Probes step (`scripts/probes_run.sh`, case `publishtensor`) run
`scripts/publish_tensor.py --data "$TENSOR"` and exit — this is exactly how
r2 was published (`family4_na025_pentad_r2_37e146384b.npz.{aa,ab,ac}`, E-044b
§2 in the log). One dispatch does build + publish:
`tensor: family4_na025_pentad_r3`, `window: recipe:f4r3-cone-5M,publishtensor`
(the recipe token is stripped and `WINDOW` becomes `publishtensor`),
`steps: 1`, `temporal_steps: 0`, `eval_every: 0`, `light_probe_every: 0`,
`runner: <a box with ≥ 100 GB disk>`, `job_timeout: 400`, `doc: "E-069 step A
— build and publish the r3 tensor (one codec step so the job is green,
nothing is trained) · params 40M · stage build-only · data
family4_na025_pentad_r3 · arch 512×12 d_z 32 patch 1 · steps 1×512 · resume
none"`. The build takes ~1.5–2 h on a fresh box (r2 took that; the r3 loop
is the same loop writing two more channels). `publish_tensor.py` chunks to
1.5 GiB parts and uploads with `curl -T` (streaming — never `--data-binary`)
as `family4_na025_pentad_r3_<sha10>.npz.{aa,ab,ac}` on `data-cache-v1`; it
is idempotent and its exit code is honest, but the CALLER wraps it in
`|| echo "::warning::…"`, so **check the release, not the step colour**:
list the assets and confirm three parts whose sizes sum to the file's size.
The full sha256 is in the job's `Record provenance` output and in
`provenance.json` on the archived branch; copy it.

**Route A2: build by hand on any machine with ≥ 60 GB free disk and ~16 GB
RAM.** Seed the inputs the way the workflow does (read the branch, it is
the executable documentation), then
`python3 ml/build_family4.py --pentad-dir <PD> --rev r3 --sst-dir <SD>`, then
`sha256sum` the output, then `GITHUB_TOKEN=… python3 scripts/publish_tensor.py --data <path>`
(token from a file, never argv). Route A2 is what to do if the workflow
cannot be edited today; it is the same code.

**Route A3: on the TPU node itself** — possible (the host has 189 GB RAM
and ~90 GB disk, and every input is public) but NOT recommended for the
first build: a node bills while it builds and the first build is the one
that has to be published to the release, which needs a GitHub token on the
node — and §6 of `ml/CLAUDE.md` forbids a PAT on a rented machine. Build
elsewhere; the TPU launcher then pulls the published chunks (release
fallback) and re-publishes the assembled file to the bucket.

**Stage it to the bucket** (the TPU path reads the bucket first, the
release second):

```
GCP_SA_KEY=<path outside the repo> python3 scripts/tpu_box.py stage \
    ml/cache/family4_na025_pentad_r3.npz \
    gs://earth-tpu-staging/tensors/family4_na025_pentad_r3_<sha10>.npz
```

**Artefacts that prove Phase A is done** (write them into
`claude/expectations.md` §2b the moment they exist):

1. **DONE 2026-09-03** — the full sha256 of `family4_na025_pentad_r3.npz` and
   its size: sha256 `fa460837fa172825ee76c8fc6fc4da75fa7b96d64519a2e2186f5c306cf03ea9`
   (fingerprint `fa460837fa`), size 5,303,481,823 bytes. Built by #535 (E-069
   Phase A — build and publish the r3 tensor, workflow run 33755174571) on
   the Ontario box (runner `gpu-box-31299601`), 23 min.
2. **DONE 2026-09-03** — the `data-cache-v1` assets: **FOUR** parts, not
   three (it chunks at 1.5 GiB and r3 is two channels larger than r2's
   three-part file) —
   `family4_na025_pentad_r3_fa460837fa.npz.{aa,ab,ac,ad}`, sizes
   1,572,864,000 × 3 + 584,889,823 = 5,303,481,823, matching item 1 exactly.
   Verified by listing the release.
3. **NOT DONE** — the bucket object
   `tensors/family4_na025_pentad_r3_fa460837fa.npz` (`gcs_size`, or
   `tpu_box.py stage` re-run reporting "already present"). The GCP service-
   account key was not available to the session that closed items 1–2, so
   this step was not attempted; the TPU launcher's release fallback
   (§8.8, now handling four parts) covers a node that starts without it —
   at the cost of the ~5-minute release-CDN pull plus the re-publish to the
   bucket instead of the ~1-minute bucket-to-bucket stage.

Then pin the sha in three places in the same commit: the workflow's r3 pull
block (§5.2), `ml/jaxport/tpu_train_cone.sh`'s `TENSOR_SHA` (§8.8), and the
expectations ledger. **Done 2026-09-03 for the first two**: the workflow's r3
pull block now carries `TPIN3=fa460837fa172825ee76c8fc6fc4da75fa7b96d64519a2e2186f5c306cf03ea9`
verbatim (§5.2). `tpu_train_cone.sh` itself keeps shipping the
`TENSOR_SHA="${TENSOR_SHA:-<the Phase-A sha256>}"` PLACEHOLDER default — that
is deliberate (§8.8's own template, `tests/test_tpu_train_cone.py` case 4):
the checked-in launcher is baked like `__BUCKET__`/`__NODE__`, never with a
live value, so a future re-build (r4, a re-chunked r3) cannot leave a stale
sha silently trusted. The real value is sed'd in at launch time, exactly as
run #536's dispatch did — `-e 's|^TENSOR_SHA=.*|TENSOR_SHA="fa460837fa...
full 64 hex"|'` alongside `__BUCKET__`/`__NODE__`/`__TPUZONE__` — with the
sha copied from item 1 above, and `TENSOR_PARTS` now defaults to `aa ab ac ad`
in the template itself (§8.8) rather than needing a per-launch override.

---

## 5 · Phase B — workflow wiring (the torch/Vast path)

Needed for Routes A1 and C1 (GPU dispatch). NOT needed for the TPU path
(§8), which is session-driven and never touches `ml-train.yml`.

**Traps first.** `workflow_dispatch` takes at most 25 inputs and the file
has exactly 25; a 26th makes EVERY dispatch in the repo 422 with a failed
run named after the file path (`ml/CLAUDE.md` §7). Every `run:` block must
stay under 21,000 characters (`tests/test_workflow_config.py`; the largest
is ~19.6k, so there is room for about one comment). New knobs go into an
existing input's VALUE or into a `# recipe-only:` key. And a recipe key the
workflow never reads as `$RECIPE_<KEY>` is REFUSED at dispatch by
`scripts/resolve_recipe.sh` — that is the feature, not an obstacle.

### 5.1 Register the tensor stem (six edit sites, one commit)

| site (line numbers as of `ac41b97`) | edit |
|---|---|
| `:60` the `tensor` input's description | add `family4_na025_pentad_r3 (E-069: r2 plus appended cur_u, cur_v — recipe f4r3, its own file name)` to the list. It is documentation AND the list a reader dispatches from |
| `:641–645` the wind seed condition | add a line `[ "${RECIPE_TENSOR:-${{ inputs.tensor }}}" = "family4_na025_pentad_r3" ] \|\|` |
| `:671–676` the `base025_na` seed condition | same line |
| `:791–792` the family-4 branch condition | same line |
| `:879–882` the build case | a second `if` block: `REV="--rev r3 --sst-dir $SD"; seed_sst` for r3 |
| `:812–838` the pinned pull | a second pull block for r3 with its own `TPIN` (after Phase A) and asset stem `family4_na025_pentad_r3_${TPIN:0:10}.npz.$sfx`. Until the pin exists, leave the pull out — an unpinned pull is a pull of nothing |

Tests to extend in the same commit: `tests/test_e034_family4.py` check 14
(the rev list) and `tests/test_family4_tensor_pull.py` (an r3 case). Run
`python3 tests/test_workflow_config.py` afterwards for the ceiling.

### 5.2 Make the cone trainer dispatchable

The Train step (~`:1128–1235`) hardcodes `python -u ml/train.py`. Encode
the switch into the `window` input, the way `unroll:`, `resume2:`, `sroll:`
and `stencil:` already are:

- `window: recipe:<name>,cone` — the `recipe:` token must come FIRST
  (`resolve_recipe.sh` matches `recipe:*`), strips it and exports the rest
  (`cone`, plus any `seed:N`) as `WINDOW` → the Train step
  branches to `python -u ml/train_cone.py --tensor "$TF" …` and the Probes
  step is SKIPPED for the codec run (there is no `Z` to probe; the cone
  codec's probe is `--velocity-probe`, run inside the trainer).
- Cone flags become `# recipe-only:` keys, each consumed exactly once in
  the cone branch as `${RECIPE_<KEY>:-<default>}`:
  `cone_l_in` (6), `cone_n_latents` (64), `cone_layers` (6), `cone_d_model`
  (256), `cone_heads` (8), `cone_d_dec` (256), `cone_dec_layers` (2),
  `cone_aux_w` (0.25), `cone_dot_queries` (256), `cone_future_lags` (`1,2`),
  `cone_eval_every` (1000), `cone_velocity_probe` (`true`),
  `cone_snapshot_ablation` (`false`), `cone_lr` (3e-4 — do NOT overload
  `lr_floor`, it already means two things). Existing inputs carry the rest:
  `steps`, `batch`, `d_z`. The seed rides the `window` input the way stage 2's
  does (`probes_run.sh:333–341`): `window: recipe:<name>,cone,seed:1` —
  the same `*seed:*` token, parsed the same way, default 0.
- Add the keys to the `# recipe-only:` comment block near `:120` — that
  block is what `resolve_recipe.sh` parses; forgetting it is a refused
  dispatch, which is the safe failure.
- Write `ml/recipes/f4r3-cone-7M.json` carrying them all, with
  `_description` and `_provenance`, and keep `f4r3-cone-5M.json` as the
  40M-class PixelMAE recipe on r3 (it is the H1 snapshot control's natural
  recipe if the control is trained through `train.py`; the cheaper control
  is `--snapshot-ablation`, which trains the L_in = 0 ConeMAE twin in the
  same job).
- The cone run's `metrics.jsonl` is already in `train.py`'s record family,
  so `publish_live_metrics.sh` and the status page need nothing.
- Archive: the job's existing `Archive metrics` step ships `metrics.jsonl`
  to `ml-metrics`; add `cone_codec.pt` to the `model-checkpoints-v1` upload
  under the tag `run-<n>__cone_codec.pt` and `velocity_probe.json` to the
  probe archive (`scripts/archive_probes.py` takes a directory — point it at
  `--out`). **Assert the effect**: after the first run, list the release
  and the `ml-metrics` branch and confirm the files are there and non-empty;
  a green archive step is not evidence (§8 of `ml/CLAUDE.md`, the 24 h token
  entry).

Run `python3 tests/test_workflow_config.py` and
`python3 tests/test_train_config_guards.py` before pushing. Dispatch nothing
until the r3 tensor has a pin OR the build-only dispatch of Route A1 is the
dispatch.

---

## 6 · Phase C — dispatch the codec and read H1

**Header** (§0d), to be the `doc` string and the EXPERIMENTS.md line:

> `E-069 · cone-native codec, inner cone lags 0–6 pentads, family reach A/B/C, +
> L_in = 0 snapshot twin for H1 · params 7.05M (ConeMAE 64 latents × 256 × 6, d_z
> 32) · stage encoder · data family4_na025_pentad_r3 · arch Perceiver 64×256×6,
> dec 256×2, Gaussian head, aux 0.25 · steps 20000×256 (+ 20000×256 for the twin)
> · resume none (fresh)`

**Route C1 — GPU box via the workflow** (after §5): `tensor:
family4_na025_pentad_r3`, `window: recipe:f4r3-cone-7M,cone`, `steps:
20000`, `batch: 256`, `d_z: 32`, `temporal_steps: 0`, `runner: <the healthy
box>`, `job_timeout: 900`. Size it: the sampler does ~0.15 s per 256
anchors on a memmap, so 20k steps is ~1 h of gather plus the network; the
plan's estimate is 2–3 h on an A100-class card, ~5–6 h with the snapshot
twin. `max_minutes` is not honoured by the cone trainer (it has no
wall-clock refit); set `job_timeout` generously.

**Route C2 — TPU** (after §8): the launcher in §8.8.

**In the first ten minutes, verify** (§2 of `ml/CLAUDE.md`): the `config`
record on the live branch reads `trainer: "cone"`, `L_in: 6`,
`n_dot_tokens: 706`, `params_M: 7.049`, `data: family4_na025_pentad_r3.npz`;
the log shows `pool certificate: 0 violations`; `gpu_util` is non-zero (or
the TPU beacon has shipped). A `config` with `L_in: 0` for the main arm, or
`n_dot_tokens` ≠ 706, is a wrong dispatch — cancel it.

**What to read when it lands** (n = 1 is a direction, never a level —
§3b): `held_out_nll` at the last eval vs step 0 (the curve must fall);
`velocity_probe.json` — `cone.cur_u.r2`, `cone.cur_v.r2` against
`snapshot.cur_u.r2`, `snapshot.cur_v.r2`, and the two deltas. H1's
prediction is 0.3–0.6 against < 0.1. Then dispatch seeds 1 and 2 (probe
claims need n ≥ 3, §3b "two seeds remain mandatory: any probe-scored
claim") before writing a level. Record all three in the E-069 entry with
the seed interval; the paper gets nothing until then.

**After the codec lands, publish**: `cone_codec.pt` to `model-checkpoints-v1`
as `run-<n>__cone_codec.pt`; `velocity_probe.json` to the probe archive;
the `metrics.jsonl` to `ml-metrics`. Update expectations (§2b: "cone_codec.pt
exists as run-<n>"), OVERVIEW (§2 item 0 → "codec landed, H1 reads …"), and
the E-069 entry.

---

## 7 · Phase D — the stage-2 arms (H2, H3)

Only after H1 has held at n ≥ 3.

**7.1 Embed the tensor with the cone codec.** Stage 2 consumes
`Z[T, P, d_z]` (float16, `temporal.CACHE_DTYPE`) over the window's ocean
pixels. `temporal.embed_everything` reads `PixelMAE`; the cone codec needs
its own pass: for every (t, pixel) build the inner cone with `ConeSampler`
and call `ConeMAE.encode`. Write `ml/embed_cone.py` (torch) mirroring
`embed_everything`'s signature and its float16-on-the-way-in convention,
batched by latitude row (the sampler caches the dot table per row), with
the resumable `.partial` and the `(codec weight hash, tensor sha)` cache
key `embed_cache_sync.py` expects. Cost: 84,405 pixels × ~3,150 bins ≈
266M encodes of a 748-token set through 64 latents — roughly 4–8× a
PixelMAE embed. Budget ~10–20 h on a 4090-class box and publish partial
chunks DURING the pass (§5.26). **A bin whose cone is incomplete** (t < L_in,
or dots off the archive's end) is embedded with those dots `valid=False` —
the model was trained with that mask semantics — and the embed file
records which bins are "short-cone" so a reader can exclude them.

**7.2 The annulus stencil in `ml/temporal.py`** — the one existing-file
change the experiment asks for. Today `--stencil` + `--ring spiral:…`
builds ONE offset set applied at every lag. The annulus arm needs a
lag-dependent set: at window position k (lag in pentads) the neighbour
offsets are `cone.outer_spiral(lat, k)` — empty for k ≤ 6, a ring from
k = 7. Implement as a new ring spelling `--ring cone:<L_in>`; the anchor
column stays at every lag; the gather becomes a ragged (per-lag) gather
padded to the max ring size with a key-padding mask in the stencil
attention. **Bit-identity control:** with `--ring spiral:111-4444-0.71-0.5`
(the E-026 spelling) the new code path must reproduce the archived head's
forward to `torch.equal` — add `tests/test_cone_roll_equivalence.py`-style
pin (that file exists for a related check; extend or sibling it). Commit
this as its own change with that test, before any arm is dispatched.

**7.3 The three arms × five seeds** (plan §6.6), all 7.6M heads (256×8,
K 144), ≤ 5k steps with selection at the held-out minimum, `--holdout-scope
window`, z-noise 0.7 — E-064b's configuration exactly except the stencil
and the Z:

| arm | Z | stencil | question |
|---|---|---|---|
| cone-annulus | cone codec | `cone:6` | H3 (and H2 with H3) |
| cone-cylinder | cone codec | E-026 spiral, all lags | H2 |
| control | run-415 (continuous d_z 32) | E-026 spiral | the bar (E-064b) |

Dispatch as `sroll:` runs so each head gets the #516 battery (per-lead
`acc`, `msss_clim` against climatology, persistence, damped persistence
and the E-066 LIM; trained/held-out longitudes separate; block-bootstrap
over (year, start)). Fit the LIM in BOTH embedding spaces as the null
(`ml/lim.py`'s embedding-space mode, if present; else the pixel-space LIM
stands as the reference and the entry says so).

**7.4 Reading.** H2 is decided on leads 1–6 for `cur_*` and `ssh`: the
cone-cylinder arm's five-seed interval against the control's. H3 on the
annulus vs cylinder pair. A single lead where the intervals separate is
not a result; the prediction names leads 1–3 and two channels. Every
number with its interval, every `#NNN` with its summary.

---

## 8 · Phase E — the JAX/TPU port

### 8.1 Why, and what a TPU number is

The TPU pool is the wall-clock-critical lane (a v5litepod-4 trains a codec
of this size several times faster than a 4090 and the session drives it
directly, no workflow), and it is the lane Chris asked for. Two rules
travel with it (`ml/plans/JAX_PORT.md` §1b/§3.4, `FGN_JAX_PORT.md` §4):

- **A TPU-trained number is a NEW TIER under `ml/CLAUDE.md` §3b.** It is
  comparable to a torch number only as a direction until a backend pair is
  measured. The first JAX cone codec buys a same-seed torch twin
  (`train_cone.py` on a GPU box, identical flags), and that pair becomes a
  new row of §3b's table.
- **Optimizer-state resume across frameworks is out of scope.** A JAX run
  resumes from its own `ckpt_latest.npz`; the exported `.pt` carries
  weights only, with `args["backend"] = "jax"`.

The port is a *reference implementation with a certificate*, in the shape
the existing `ml/jaxport/` files have (`README.md` there): NNX mirror →
two-way converter with a refusal contract → parity gates with
pre-registered tolerances → optax trainer emitting the same records → a
launcher cloned from `tpu_train_field.sh`. **A gate that fails is a finding
about the port, never a tolerance to widen** (G5c's history: it failed,
three seeds showed it systematic, the fix was the init, not the band).

### 8.2 Files to create

| file | contents |
|---|---|
| `ml/jaxport/cone_models.py` | `ConeMAEJax(nnx.Module)` + `CoordEncJax` + `CrossBlockJax`; reuses `models.MultiHeadAttention`, `models.TransformerEncoder`, `models.gelu_exact`, `models._nnx_data` |
| `ml/jaxport/cone_convert.py` | `load_cone(state_dict, model)`, `export_cone(model) → state_dict`, `export_cone_pt(model, args, path, **extra)`, `cone_from_torch(torch_model)`, `cone_from_ckpt_jax(blob)` — same `_Consumer` / `_Emitter` bookkeeping as `convert.py` (import them; do not copy) |
| `ml/jaxport/train_cone.py` | the optax trainer; imports `load_data`, `admissible_bins`, `draw_anchors`, `fold_labels`, `kfold_r2` and `ConeSampler` from the torch-side modules (the CPU torch wheel is on the node for exactly this — G5c's finding) |
| `tests/test_jaxport_cone.py` | gates C1–C8 (§8.7); plain `python3 tests/test_jaxport_cone.py`, CPU, fp32, no network — the `jaxport` convention |
| `ml/jaxport/tpu_train_cone.sh` | the launcher (§8.8) |
| `ml/cone_codec.py` (ADDITIVE edit) | `forward_given(b, plan, chan_mask, dot_mask, dot_idx)` and `draw_dot_queries(b, plan, dot_mask) → (idx, sel)`, so the randomness can be handed to both frameworks; `forward` keeps its numbers (a test asserts it) |

### 8.3 Module-by-module mapping, with every trap

The torch state_dict has 99 keys at any geometry (listed by
`ConeMAE(...).state_dict().keys()`). The mapping:

| torch | NNX | trap |
|---|---|---|
| `val_proj`, `dot_proj`, `ctx_proj`, `coord.proj`, `to_z`, `q_proj`, `z_proj`, `lat_proj`, `head`, `*.linear1/2`, `*.mlp.0/2` (all `nn.Linear`) | `nnx.Linear`; `kernel = weight.T` | **transpose every linear** (`[out,in]` → `[in,out]`) |
| `chan_emb` (`nn.Embedding`) | `nnx.Embed`; `embedding = weight` | init differs (N(0,1) vs 1/√d) — init from torch (§8.4) |
| `mask_tok`, `miss_tok`, `cls_tok`, `query_tok`, `latents`, `pool_q` (`nn.Parameter`) | `nnx.Param` | shapes `[d]`, `[n_latents, d]`, `[1, 1, d]` kept verbatim |
| `coord.freqs` (buffer, NOT in the state_dict) | recomputed: `2.0 ** arange(n_fourier)` | the converter must not look for it, and the test must check it is identical |
| `cross.attn`, `pool`, `dec.<i>.attn`, `encoder.layers.<i>.self_attn` (`nn.MultiheadAttention`) | `models.MultiHeadAttention`; `in_proj_weight [3d, d]` sliced **q, k, v in that order** into `q_proj/k_proj/v_proj` (transposed), `in_proj_bias` likewise; `out_proj` as a linear | a wrong slice trains and produces numbers |
| `cross.ln_q/ln_kv/ln_m`, `ln_pool`, `ln_out`, `*.norm1/2` (`nn.LayerNorm`) | `nnx.LayerNorm(d, epsilon=1e-5)` | **eps 1e-5, not Flax's 1e-6** |
| `CrossBlock.mlp` = `Linear → GELU → Linear` | `gelu_exact` (erf) | `jax.nn.gelu` defaults to tanh — 1e-3 off |
| `encoder` = `nn.TransformerEncoder(nn.TransformerEncoderLayer(d, h, 4d, batch_first, norm_first, dropout 0), n)` | `models.TransformerEncoder(d, h, n, 4d)` | activation is **RELU** (no `activation=` argument in `cone_codec.py`), **no final norm** |
| `key_padding_mask` (bool [B, L], True = ignore) | `mask = kpm[:, None, None, :]` into `MultiHeadAttention(..., mask=)` — that class already treats a bool mask as "True = masked out" and `jnp.where`s to −inf | torch's semantics exactly; no row is ever fully masked (cls/ctx/patch tokens are always valid), so no NaN guard is needed — assert that in the test rather than adding one |
| `torch.where(vis, vt, 0) + where(hid, mask_tok, 0) + where(mis, miss_tok, 0) + base` | `jnp.where` with the same three exclusive conditions | `obs & ~mask` / `obs & mask` / `~obs` — exclusive by construction; keep the ORDER of additions (fp32 summation order is part of parity at 1e-6) |
| `logvar.clamp(-8, 8)` | `jnp.clip` | — |
| `signed_log(v) = sign(v)·log1p(|v|)` | same | negative lag days (future queries) go through it; do not "simplify" to log1p |
| `nll_gauss` | same formula | keep `0.5 * (log 2π + logvar + (t − μ)² · exp(−logvar))` in that order |
| `_masks` (torch RNG) | host-side numpy in the trainer (§8.6); `forward_given` in both frameworks takes the arrays | two RNGs cannot agree — the gate passes the DRAW, not the seed |
| `_query_sets` topk over random scores | `draw_dot_queries` returns `idx [B, k]`, `sel [B, k]`; both frameworks gather with the given `idx` | the torch `forward` (unchanged) still uses `topk`; the gate compares `forward_given` to `forward_given` |
| `loss = nll_z + aux_w · nll_latent` | same | `wsum` clamp at 1e-6 |

### 8.4 Initialisation: from the torch module, always

`nn.Embedding` is N(0, 1); `nnx.Embed` is std ≈ 1/√d. `README.md:105–118`
measured the consequence on stage 2 (flax-initialised heads read 1.46/1.87/
1.68 against torch's 0.71/0.87/0.77). So the JAX cone codec is initialised
by constructing `ConeMAE` under `torch.manual_seed(seed)` and converting
(`cone_from_torch`), exactly as `field_head_from_torch` does. The trainer
does this at step 0 on a fresh run; a resumed run loads `ckpt_latest.npz`.
The seed in the `config` record is the torch seed.

### 8.5 The converter contract

`load_cone(sd, model)`: consume every key above through `_Consumer`; on
`finish()` refuse a partial load naming missing AND unconsumed keys.
`export_cone(model)`: the mirror, through `_Emitter`, producing a
state_dict `torch.load`-able into `ConeMAE` with `strict=True`.
`export_cone_pt(model, args, path, chan_names, norm, step, arm, L_in, params)`
writes the SAME blob shape `train_cone.py`'s `save()` writes, so
`velocity_probe`, the future `embed_cone.py`, and any eval script read it
unchanged; `args["backend"] = "jax"` is the one mark. `cone_from_ckpt_jax`
mirrors `ConeMAE(**geometry from args)` + `load_cone` with the same `.get()`
defaults the torch side uses.

### 8.6 The trainer `ml/jaxport/train_cone.py`

Mirror `ml/train_cone.py` flag for flag; REFUSE any flag it cannot honour
(the `jaxport` convention — `--eval-every` refuses in `train_stage1.py`,
five flags refuse in `train_stage2.py`), never accept-and-ignore.

- Optimiser: `optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr_schedule, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.01))`.
  torch's `clip_grad_norm_(1.0)` then `AdamW` step is that order. Schedule:
  `optax.cosine_decay_schedule(init_value=lr, decay_steps=steps, alpha=0.0)`
  — `CosineAnnealingLR(T_max=steps)` stepped AFTER `opt.step()` means update
  s uses `lr · ½(1 + cos(π (s−1)/steps))`, which is optax's `count = s − 1`.
  Gate C3 checks one step of each; C4 checks 300 in a band.
- Masks: a numpy `Generator` seeded from `--seed` draws `chan_mask` (Bernoulli
  per channel with `chan_drop_p`), the lag band (`on ~ p`, `l0 ∈ {1,2,3}`,
  `lag_days ≤ 5·l0`), the sector (`th0 ~ U(0, 2π)`, `remainder(atan2(dx, dy) − th0, 2π) < π/2`),
  and `dot_idx` (a random subset of `dot_mask & obs & valid`, capped at
  `n_dot_queries`, padded with `sel=False`). Same distributions as
  `ConeMAE._masks` / `_query_sets`; the stream is not shared and the gate
  does not pretend it is.
- Step function: `nnx.jit` over `(model, opt_state, batch, masks) → loss,
  terms`; the batch is padded to a FIXED `N` (the sampler's `n_dots` is
  per-row; pad to the max over all rows once, so XLA compiles once — a
  per-batch `N` retraces every step, the `embed.py` lesson). `Q` (queries)
  is also fixed: `C + C·F + n_dot_queries`.
- Records: `ml/train.py`'s family, identical keys to §2.4, plus
  `"backend": "jax"` in `config`. `wall_s` restarts at 0 on a new node —
  write a `{"resumed": {"at_step": s, …}}` record BEFORE the `config`
  record on resume (the field trainer's convention; status.html keeps
  the seam).
- `results.json` written at every eval with `in_progress: true`
  (temp sibling + `os.replace`); the final write drops the marker (§5.25).
- Checkpoint every `--ckpt-every` steps: `ckpt_latest.npz` = NNX state +
  optax state + step + the numpy RNG state, written atomically; alongside
  it `cone_codec.pt` via `export_cone_pt` (torch-loadable, so the probe
  and the eval ladder need not wait). `--resume` loads the `.npz` and
  continues bit-identically (gate C8).
- Velocity probe: encode with the `cur_*` channels hidden (the `chan_mask`
  / `dot_mask` path — `forward_given`'s masks, no loss), then
  `train_cone.kfold_r2` on the host (it imports `probe_kfold.kfold_r`,
  which imports torch — the CPU wheel is present). Same `velocity_probe.json`
  shape. The snapshot twin (`--snapshot-ablation`) is the same trainer at
  `--L-in 0`.
- Certificate before step 0: `sampler.certify` — a refusal here costs
  nothing, at step 5,000 it costs the node.

### 8.7 The gates — `tests/test_jaxport_cone.py`, tolerances pre-registered here

Build one pair on identical weights: `ConeMAE` (torch, smoke geometry
`n_chan 8, d_model 32, heads 4, latents 8, layers 2, d_z 8, d_dec 32,
dec_layers 2`) under `torch.manual_seed(0)` → `cone_from_torch`. One batch
from the smoke tensor (`train_cone.smoke_tensor`, `ConeSampler`, 16
anchors incl. at least one with invalid dots). Print every measured
difference; fail only against the tolerance.

| gate | what | tolerance |
|---|---|---|
| C1 forward | `tokens` (both outputs), `encode` (z and latents), `query_tokens`, `decode_from_z`, `decode` on identical inputs and identical given masks | ≤ 1e-4 (expect ≤ 1e-6, the G1 experience) |
| C2 loss | `forward_given` loss and every `terms` scalar, aux on and off | ≤ 1e-5 |
| C3 one step | SGD lr 1e-2: every parameter ≤ 1e-6; AdamW (the real chain): ≤ 2e-5 overall, ≤ 1e-6 where \|g\| > 1e-6 — G4b's four-way statement | as stated |
| C4 band | 300 steps of the JAX trainer vs `python3 ml/train_cone.py --smoke --steps 300` as a SUBPROCESS on the same smoke tensor and seed; compare the final `held_out_nll` | ratio in [0.8, 1.25], and both below their step-0 value (a band, not equality — the mask streams differ) |
| C5 round trip | `export_cone(load_cone(sd))` state_dict identical to `sd` (`torch.equal`, dtype included); re-encode through the exported `.pt` loaded into torch `ConeMAE` | identical / ≤ 1e-5 |
| C6 refusal | a state_dict missing one key, and one with an extra key, both raise and NAME the offender | must raise |
| C7 existence | perturb an invalid dot's value in the JAX batch → z bitwise equal (`np.array_equal`) — the torch smoke test's check, on the port | exact |
| C8 resume | train 20 steps, checkpoint, train 20 more; vs 40 straight | bitwise on the state |
| C9 `forward` unchanged | torch `forward(b, plan)` with `generator` seeded g equals `forward_given` fed the masks drawn from the same seeded g | exact |
| C10 production geometry | the pair REBUILT at the dispatch geometry (42 real r3 channels, d_model 256, 64 latents × 6, 706 dots): parameter identity on both sides against §0's number, the sampler's dot count against `cone.budget()`, forward and loss at C1/C2's tolerances, **gradient** parity over every element, the depth axis proven live, and the `L_in = 0` snapshot arm | identity · 1e-4 · 1e-5 · 1e-6 (grads) |

**Why C10 exists, and why it is the one that matters.** C1–C9 all run at the
smoke geometry — 8 invented channels, 66,090 parameters, no depth channel, no
family mix. Three things can only be wrong at the dispatch size: a parameter
the converter never sees because the small geometry collapses two dimensions
onto the same number; a coordinate encoding whose conditioning changes with
the real reaches (to 907 km) and the real depth ladder (to 1,900 dbar); and
the GRADIENT, which C3 only ever probes through one optimiser step on a model
2% the size. A forward can agree while a backward disagrees, and only the
backward moves weights — so C10 is the gate that actually says a TPU run
trains the same model.

Run time: ~5 minutes on CPU (C4's two subprocesses and C10's 7.05M-parameter
gradient dominate). If any gate fails, the diff is the finding; write it into
the test's docstring like `models.py` does.

**MEASURED 2026-09-02**, all ten green (`python3 tests/test_jaxport_cone.py`):
C1 tokens 1.26e-05, z 2.09e-07; C2 loss 9.54e-07; C3 SGD 1.19e-07, AdamW
1.17e-05 / 2.38e-07 where |g| > 1e-6; C4 torch 2.0367 → 1.7528 against jax
2.0452 → 1.7527, ratio 1.0000; C5 99/99 keys `torch.equal`; C6 both refusals
name the offender; C7 exact; C8 373 leaves bitwise across the seam; C9 exact;
**C10 at the dispatch geometry — 7,048,994 parameters on both sides, 706 dots
= `budget()`, tokens 2.03e-05, z 4.47e-07, loss 2.38e-07, gradient max|Δ|
6.56e-07 over 5,219,652 elements with |g| > 1e-6, global gradient norm
12.4837 both sides (rel 4.9e-07), depth 10 vs 1,900 dbar moves the encoding
by 0.82, snapshot arm z 3.05e-07.** C1's 1e-5 on tokens is `CoordEnc`'s own
condition number (the top Fourier band multiplies a one-ULP `log1p`
difference by 128) and does not propagate; the reasoning is in the test's
docstring. Two findings came out of writing them: the converter aliased 68 of
99 tensors to torch storage (fixed, `convert._Consumer.get` copies), and
`export_cone` returns numpy that `load_state_dict` refuses — `export_cone_pt`
is what makes the loadable artefact (its docstring now says so).

### 8.8 The launcher `ml/jaxport/tpu_train_cone.sh`

Clone `ml/jaxport/tpu_train_field.sh` (the most complete one; sections
0–9 in its header) and change ONLY what the cone needs. **Sed the previous
run's own file for a relaunch, never the pristine template, and `diff` the
knob blocks** (`SPOT_LEDGER.md` 08-28: a template rebake reverted every
knob and spent 62 min re-embedding a published Z).

Keep verbatim, they are the safety kit the spot-first rule depends on:
the `__BUCKET__` / `__NODE__` / `__TPUZONE__` placeholders (the create
command refuses an unbaked file); the EXIT trap → `ship_final` →
`upload_log` → `self_delete` (asserts the 404); the **boot beacon** at the
banner (`upload_log`, then a 3-minute shipper retired by `kill
"${BEACON_PID}"` when the 10-minute shipper arms); the `resolved knobs`
echo; the progress watchdog on the SHIPPED `ckpt_latest.npz` (`STALL_MIN`);
`MAX_HOURS`; `ship_file`'s change detection; `gcs_put` with `curl -T`;
`gcs_size`'s transport-retry; `gcs_publish` via `.uploading` + size check;
the disk guard with the `/dev/shm` remount; the deps section
(`systemctl stop unattended-upgrades`, venv, `jax[tpu]`, `flax` ≤ 0.10 on
the Ubuntu 22.04 / Python 3.10 image, `optax`, CPU torch wheel); the code
clone at a pinned `GIT_SHA` (empty is wrong for a long train); the bucket-
first / release-fallback tensor path with `sha256sum` refusal and
`--retry 3`; resume by the same `__NODE__`; the metadata-server token for
every bucket and delete call (no key on the node).

Change:

```
# knobs — the cone codec
TENSOR_NAME="${TENSOR_NAME:-family4_na025_pentad_r3}"
TENSOR_SHA="${TENSOR_SHA:-<the Phase-A sha256>}"        # REQUIRED, refuse if empty
TENSOR_PARTS="${TENSOR_PARTS:-aa ab ac ad}"              # r3 is FOUR parts, not three
GCS_TENSOR="${GCS_TENSOR:-tensors/${TENSOR_NAME}_${TENSOR_SHA:0:10}.npz}"
EST_TENSOR_BYTES="${EST_TENSOR_BYTES:-11500000000}"
STEPS=20000 BATCH=256 LR=3e-4 SEED=0 D_Z=32
D_MODEL=256 HEADS=8 N_LATENTS=64 LAYERS=6 D_DEC=256 DEC_LAYERS=2
L_IN=6 FUTURE_LAGS="1,2" AUX_W=0.25 DOT_QUERIES=256
HOLDOUT_YEARS="2009,2017,2023" EVAL_EVERY=1000 CKPT_EVERY=1000
VELOCITY_PROBE=1 SNAPSHOT_ABLATION=0 TAG=""
# no Z, no codec asset, no pixels object: the cone codec is stage 1
```

Drop the Z and pixels staging (there is no Z), the field `MODE=verify`
leg's diffusion knobs, and the `field_latest.pt` name (`cone_codec.pt`).
Keep a `MODE=verify` leg: 300 steps at the real geometry on the real
tensor, `VERIFY_MAX_HOURS=2`, and it must PRINT `measured: <s/step>` so
`STEP_EST_S` stops being a stand-in. Stall arithmetic for the cone,
first-run values (state them in the header the way the field launcher
does): `SETUP_EST_MIN=12`, `STAGE_MBPS=200`, `RELEASE_MBPS=40`, tensor
11.5 GB → ~1 min from the bucket / ~5 min from the release, the anomaly
scratch copy ~36 GB written on the node (`NEED_GB=120` covers it, `/dev/shm`
at 170 G if the disk does not), `STEP_EST_S` **unknown — take 0.5 s until the
verify leg measures it** (the host gather bounds it; the sampler's numpy
gather runs on the 112-CPU host and was 0.15 s/256 anchors on a 4-core
sandbox), `CKPT_EVERY=1000` → `t_first_ship ≈ 12 + 1 + 1000·0.5/60 + 10 ≈ 31
min` against `STALL_MIN=90` — clears. If the verify leg reads > 4 s/step,
the gather is starving the chip: batch anchors by latitude row in the
sampler call (it already groups by row) and prefetch one batch on a thread,
the `tpu_train_s2.sh` pipeline-knob pattern (`GATHER_WORKERS`, `PREFETCH`).

Bucket layout the launcher writes (`runs/<node>/`): `metrics.jsonl`,
`results.json` (with `in_progress`), `ckpt_latest.npz`, `cone_codec.pt`,
`velocity_probe.json` (at the end), `verify_report.json` (verify mode),
`logs/<STAMP>.txt`.

`scripts/tpu_status_mirror.py` copies `<node>/metrics.jsonl` into
`ml-live-tpu`; `status.html`'s `stage2Chart`/`fieldChart` will draw NOTHING
for `train.py`'s family in the TPU card (they key on `stage2_*` and
`field_*`). **Add a `coneChart` (or teach `stage2Chart` the `{step, loss_rec}`
+ `held_out_nll` pair) in the same change that ships the launcher** — §0d.
The Vast path renders it already because the run-page code path
(`parseJsonl` → the codec chart) is the one `train.py` uses.

### 8.9 Launch procedure (session-driven; no workflow)

Credentials: the SA key lives in the claude.ai project doc
[`claude/gcp-tpu-access.md`]; write it to a file OUTSIDE the repo tree for
the life of the session (`tpu_box.py` refuses a key path inside the tree),
export `GCP_SA_KEY=<path>`, never put the key on a command line, never on a
node. Project `earth-tpu-blauewelt`, bucket `gs://earth-tpu-staging`, SA
`tpu-runner@earth-tpu-blauewelt.iam.gserviceaccount.com`. Only
`us-central1-a`, `us-east5-a/b/c`, `us-west1-c` serve `v5litepod-4`;
`us-west4-a/b` are on the ladder because they have served it under spot.

1. **Bake** the startup file: `sed -e 's|__BUCKET__|earth-tpu-staging|' -e 's|__NODE__|e069-cone-s0|' -e 's|__TPUZONE__|<zone>|' -e 's|^STEPS=.*|STEPS=20000|' … ml/jaxport/tpu_train_cone.sh > /tmp/e069.sh` — sed every knob you intend, then `grep -n '^[A-Z_]*=' /tmp/e069.sh` and READ it. A startup script inherits no environment.
2. **Spot first, then the ladder** — `us-west1-c · us-west4-a/b · us-central1-a · us-east5-a/b/c`:
   `python3 scripts/tpu_box.py create e069-cone-s0 --zone <z> --spot --accelerator-type v5litepod-4 --startup-file /tmp/e069.sh`.
   Read the op error with `SPOT_LEDGER.md`'s decoder: `Reservation not found` = spot dry, next zone; `insufficient capacity` = dry both kinds; `429 Limit: 4` = your own node holds the zone's quota (one spot v5litepod-4 per zone). On-demand only when every zone refuses spot; **append the ladder walk and its cost to `ml/SPOT_LEDGER.md` in this session.**
3. **The lemon guard** deletes a born-UNHEALTHY node and exits non-zero; re-run `create` for a fresh draw.
4. **Within ~6 minutes of READY**, `runs/e069-cone-s0/logs/` must hold a log (the beacon). If not: `tpu_box.py delete e069-cone-s0 --zone <z>` and redraw. Do not wait for `STALL_MIN` — it lives inside the script that did not run.
5. **Read the `resolved knobs` line** in the shipped log before anything else. The config is the experiment.
6. **Monitor** on status.html's TPU cards (15-minute mirror) and by `gcs_size` on `ckpt_latest.npz`; a new checkpoint object every ~`CKPT_EVERY · s/step` is health, a fresh log alone is not.
7. **At the end** the node self-deletes; confirm with `tpu_box.py list` (nothing billing) and `delete` asserts 404 if you have to do it by hand. Every session ends with `list` reading empty.
8. **Harvest**: `gcs_get runs/e069-cone-s0/{cone_codec.pt,velocity_probe.json,metrics.jsonl,results.json}` (or the mirror's copy of `metrics.jsonl`); publish `cone_codec.pt` to `model-checkpoints-v1` as `tpu-e069-cone-s0__cone_codec.pt`; archive `metrics.jsonl` to `ml-metrics` under the node name; record in EXPERIMENTS.md (§3b tier note: "TPU, n = 1, torch twin owed"), OVERVIEW, expectations.
9. **The torch twin**: dispatch `train_cone.py` with identical flags and seed on a GPU box (§6, Route C1). The pair's `velocity_probe` deltas and final `held_out_nll` are the first cross-backend row for the cone tier.

### 8.10 Delegation plan (Opus subagents, the session verifies)

Four subagents can run in parallel on disjoint files, then one integration
pass:

1. `cone_models.py` + the additive `forward_given` / `draw_dot_queries` in
   `cone_codec.py` + gate C9 (torch-only, proves `forward` is unchanged).
2. `cone_convert.py` + gates C1, C2, C5, C6, C7 (needs 1's module API — give
   both agents this document's §8.3 table as the interface).
3. `train_cone.py` (JAX) + gates C3, C4, C8.
4. `tpu_train_cone.sh` + the status-page `coneChart` + `bash -n` +
   `tests/status.spec.js` pin for the new record family.

The session then runs `python3 tests/test_jaxport_cone.py` and the 42 cone
tests, reads the diff of `cone_codec.py` (additive only), and commits.
Subagent reports are intentions until the tests have been run here.

---

## 9 · Verification checklist — the artefact each phase leaves behind

| phase | done when | how to check |
|---|---|---|
| local | 42 cone tests green; smoke prints the §3 lines | run them |
| A data | sha256 + size recorded; 3 release assets; 1 bucket object | list the release; `gcs_size`; expectations §2b updated |
| B workflow | `tests/test_workflow_config.py`, `test_e034_family4.py`, `test_family4_tensor_pull.py`, `test_train_config_guards.py` green; a dispatch with `window: recipe:f4r3-cone-7M,cone` resolves without a `$RECIPE_` refusal | dispatch a `steps: 1` dry run on `ubuntu-latest` |
| C codec | `config` record with `trainer: cone`, `L_in: 6`, `n_dot_tokens: 706`; certificate 0; `cone_codec.pt` on the release; `velocity_probe.json` archived; three seeds before any level | open the files |
| D stage 2 | annulus ring bit-identity test green; 15 `sroll:` bundles archived with per-lead tables; LIM null in both spaces | `probes-<n>.json` on `ml-metrics` |
| E port | C1–C9 green on CPU; verify leg's `measured: s/step` in the bucket log; node deleted (list empty); twin dispatched; §3b row added | `python3 tests/test_jaxport_cone.py`; `tpu_box.py list` |

---

## 10 · Cost and time (estimates; the verify leg replaces them with measurements)

| item | wall clock | $ |
|---|---|---|
| r3 build + publish (Route A1) | ~2 h on a Vast box | ~$1 |
| codec + snapshot twin, GPU | ~5–6 h | ~$2–3 |
| codec, TPU spot v5litepod-4 | ~1–3 h (gather-bound; unmeasured) | ~$3–7 spot, ~$5–15 on-demand |
| three seeds of H1 | 3× the above | |
| cone embed of the tensor (7.1) | 10–20 h, 4090-class | ~$3–6 |
| 15 stage-2 heads with the battery | 15 × ~2 h + rolls | ~$15–25 |
| JAX port (engineering) | one session with four subagents | Fable/Opus quota, no GPU |

---

## 11 · Glossary (for a reader new to the programme)

- **anchor** — the (t, y, x) cell an embedding is computed for.
- **inner cone / outer cone** — the part of the dependency cone the codec
  reads (lags 0–6) / the part stage 2 reads over embeddings (lags 7–143).
- **dot** — one (channel, lag, Δy, Δx) sample in the inner cone; a "dot
  token" is its embedding.
- **family A/B/C/rg** — channel classes by physics: wind stress; currents,
  SSH and the Argo column; SST and MLD; the depth channels.
- **z** — the d_z = 32 embedding; **latents** — the Perceiver's 64 × 256
  working set, never handed downstream.
- **mask_tok / miss_tok** — a value WE hid vs a value the data never had.
- **window scope** — the holdout rule: every bin a training anchor's cone
  reads must be a training bin (`c25f6ff` generalised to the dot set).
- **the #516 battery** — the rolled-skill protocol of the reset: per-lead
  `acc`, `msss_clim`, four baselines including the LIM, block bootstrap.
- **snapshot twin** — the same ConeMAE at `L_in = 0` (only the lag-0 patch),
  H1's control.
- **beacon** — the launcher's first log upload, within ~3 minutes of start,
  which is what makes "no object in 6 minutes ⇒ zombie" a valid verdict.

---

## 12 · Links

- [E-069 plan](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E069_cone_codec.md)
- [The generic-embedding proposal](https://blauewelt.github.io/earth/docs.html?f=ml/figures/geofm_survey/GENERIC_EMBEDDING_INPUTS.md)
- [The survey deck with notes (PDF)](https://github.com/blauewelt/earth/blob/main/ml/figures/geofm_survey/geospatial-representation-models-with-notes.pdf)
- [E-069 in the experiment log](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-069)
- [The overview](https://blauewelt.github.io/earth/docs.html?f=ml/OVERVIEW.md)
- [ml/CLAUDE.md — the standing rules](https://blauewelt.github.io/earth/docs.html?f=ml/CLAUDE.md)
- [The JAX port plan](https://blauewelt.github.io/earth/docs.html?f=ml/plans/JAX_PORT.md)
- [jaxport README (tiers, gates, how to run them)](https://blauewelt.github.io/earth/docs.html?f=ml/jaxport/README.md)
- [E-069 handover — this file](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E069_HANDOVER.md)
- [TPU access](https://blauewelt.github.io/earth/docs.html?f=ml/plans/TPU_ACCESS.md)
- [The spot ledger](https://blauewelt.github.io/earth/docs.html?f=ml/SPOT_LEDGER.md)
- [The status page](https://blauewelt.github.io/earth/status.html)

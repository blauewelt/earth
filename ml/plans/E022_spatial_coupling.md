# E-022 · Spatial coupling: predict a pixel from its neighbourhood

**Status: PLAN — nothing below has been dispatched.** Written 2026-08-13 by
the working session at Chris's request, deliberately detailed enough that a
different (weaker) model, or a fresh session with no memory of this one, can
implement and run it without asking questions. Read it in order; the design
decisions in §3 are settled — implement them, don't relitigate them.

Chris's ask, verbatim intent: *"implement predictions that depend on the
neighboring pixels … either 9 pixels (3×3) or a 5×5 … as an optimization
leave out the second diagonal, so you'd have something more circular:
2 in each direction but only one on the diagonal. As a baseline, the eval
needs to involve rolling forward all pixels that contribute to the AMOC
current (from the Gulf of Mexico to northern Europe, and back)."*

---

## 0 · What you need to know before touching anything

- **Governing rules:** `ml/CLAUDE.md` (dispatch discipline, fleet lore,
  security). Rule 1: write the EXPERIMENTS.md entry AT DISPATCH, hypothesis
  first. E-021 broke this rule and had to record the violation; don't repeat.
- **What the system is:** `docs/ML_BASICS.md`. Short version: a frozen 41M
  PixelMAE codec (run-62, weight hash `6c52f0687b`) maps each pixel-month to
  a 64-d embedding z; a stage-2 temporal transformer (`ml/temporal.py`)
  predicts z_{t+1} per pixel from that pixel's K=24-month history. Today it
  has **zero cross-pixel coupling** — the only spatial information is a
  static per-pixel identity added to every step. E-021/E-021b measured the
  consequences: rolls decay to a seasonal limit cycle, hindcast "skill" is
  memorisation, the fan is ~5× under-dispersed.
- **The tensor:** `family3_na025` — 0.25°, window lat 0..70 N, lon
  −100..+20 E, grid H=281 × W=481, T=516 months (1982-01..2024-12), C=39
  channels, `chan[0] == "cur_speed"`. 84,405 ocean pixels. sha-pinned
  fingerprint `adcbe700fb`, seeded to every box from `data-cache-v1`.
- **The Z cache:** `embed-cache-v1` release, assets
  `Z_6c52f0687b_adcbe700fb.npy.{aa,ab,ac,ad}` — f16, shape
  [516, 84405, 64]. Boxes usually have a local copy matching
  `ml/cache/Z_*_6c52f0687b_adcbe700fb.npy`.
- **The baselines that already exist (do NOT retrain them):** the three
  E-017 heads `e017_u1_s0..2__temporal.pt` on `model-checkpoints-v1`
  (576/8 ≈ 32M params, K=24, U=1, seeds 0/1/2). Their rolled numbers, from
  #217: AUC(msss_damped, h=1..12) = 0.643/0.644/0.645; amp at h=12 ≈ 0.805;
  AMOC truefit h1–3/h4–6/h7–12 = 0.458/0.353/0.463.
- **Seed noise floors, measured:** nowcast probe sd ≈ 0.12 (E-010) — a
  nowcast difference under ~0.25 at one seed each means nothing. Rolled AUC
  sd ≈ 0.001–0.005 (#217) — tight; still run 3 seeds per arm.

## 1 · Hypothesis, falsifiers, and the physics stated up front

**Hypothesis.** Feeding each pixel's *neighbourhood* of embeddings (not just
its own) lets the stage-2 model represent local transport/diffusion, which
should show up as (a) higher rolled field skill at h=1..12, (b) slower
amplitude decay, and (c) better held-out-year tracking in the long hindcast.

**Primary metric** (the one that decides): rolled **AUC(msss_damped,
h=1..12)** over the AMOC corridor (§3.6) and over the full window, three
seeds per arm, against the E-017 stencil-1 band **0.643–0.645**.
**Falsifier:** if no spatial arm's 3-seed mean exceeds 0.645 + 3×(pooled
seed sd), spatial coupling at this resolution/cadence does not help and the
axis closes. Secondary metrics: amp retention at h=12 (baseline 0.805);
AMOC truefit bands (baseline above); held-out-year median r in the 240-month
hindcast (baseline +0.34, from E-021b).

**Pre-registered physics caveat — write this into the entry so the result
is interpretable either way.** One roll step is ONE MONTH. Advection reach
per step, in 0.25° cells (~27 km at these latitudes):

| flow | speed | cells/month |
|---|---|---|
| Gulf Stream / Florida Current core | ~1–2 m/s | ~100–200 |
| North Atlantic Current | ~0.2–0.5 m/s | ~20–50 |
| deep western boundary current | ~2–10 cm/s | ~2–10 |
| interior gyre / Rossby waves | ~1–5 cm/s | ~1–5 |

A 3×3 stencil reaches 1 cell/step; the 13-point reaches 2. **No small
stencil can represent boundary-current advection at monthly cadence** — that
information moves 50–200× faster than the stencil. What the stencil CAN
capture is slow interior dynamics, deep flow, and wave propagation, plus
smoothing/consistency effects. So the honest expectation is a modest gain
concentrated away from the Gulf Stream core, and a null here does NOT rule
out spatial coupling at daily cadence or with a global operator — it rules
out *local monthly* coupling. Say so in the RESULT whatever happens.

## 2 · The two arms and their input shapes (Chris's spec, made exact)

Stencil offsets, `(dy, dx)` in grid cells, **centre first, order fixed**:

```python
STENCILS = {
    1:  [(0, 0)],
    9:  [(0, 0), (-1, -1), (-1, 0), (-1, 1), (0, -1),
         (0, 1), (1, -1), (1, 0), (1, 1)],                      # 3x3
    13: [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1),              # cardinals 1
         (-1, -1), (-1, 1), (1, -1), (1, 1),                    # diagonals 1
         (-2, 0), (2, 0), (0, -2), (0, 2)],                     # cardinals 2
}
```

`13` is Chris's "circular" shape: two steps in each cardinal direction,
one on each diagonal — the 5×5 with its outer ring trimmed to the classic
13-point stencil. Do not add (±2,±1) "knight" cells; the shape above is the
spec. Stencil **1 must reproduce today's model exactly** (see §4.2).

**No longitude wrap.** The window is regional (−100..+20 E), not global.
A neighbour off any edge of the window, or on land, is MISSING: its z is
zero-filled and its observed-flag is 0. (`model.py`'s `gather_px` wraps x
for the GLOBAL codec grid — do not copy that behaviour here.)

## 3 · Settled design decisions

1. **Input construction.** Per time step, the model input is the
   concatenation of the stencil's z vectors (missing → zeros) plus the month
   features: `[S·d_z + 2]` where S ∈ {1, 9, 13}, d_z = 64. The per-cell
   observed flags are STATIC geometry (land/edge does not change with time),
   so they go into the static context once, not into every step:
   `static_ctx = [static-z (d_z) | lat/90, lon/180 | obs_flags (S)]` →
   `[d_z + 2 + S]`. For stencil 1 both layouts reduce to today's exactly
   (S·d_z+2 = 66, d_z+2+1 → see §4.2 back-compat note).
2. **Output unchanged**: the model still predicts the CENTRE pixel's
   z_{t+1}. Loss, persistence baselines, probes all read centre z — most of
   temporal.py doesn't change.
3. **U=1 only.** `--stencil > 1` with `--unroll > 1` must be REFUSED with a
   clear message: the training-time unroll feeds back predictions, and a
   random pixel batch does not contain the neighbours' predictions. (Unroll
   is also dead on merit: E-010 + E-020 closed it on every axis.)
4. **One gather helper.** temporal.py currently gathers (Zt, Mt) windows in
   FIVE places (train batch ~1067, unroll futures ~1080, monitor ~1151,
   eval ~1336/1341, light probe ~1262/1388). All of them must go through a
   single `gather_stencil(Zt, t_starts, p_idx, NBR, K)` helper so the
   stencil logic exists once. This is the highest-risk part of the change;
   a missed call site trains on stencil inputs and evaluates on centre-only,
   which FAILS SHAPES loudly (good) — but only if the helper owns them all.
5. **The eval rolls the FULL window, not a corridor.** With a stencil, a
   correct T-step roll of a region needs the region plus a T·reach halo; the
   clean solution is to roll all 84,405 ocean pixels (it is exact for the
   per-pixel baseline too, since that factorises). Feasibility is measured,
   not guessed: the state that must persist per step is z for all P
   ([84405, 64] f32 ≈ 21 MB) and the K-window ([84405, 24, 64] ≈ 500 MB) —
   fine on a 24 GB 4090 with chunked forwards (§6).
6. **The corridor is for SCORING, and it is data-derived, not hand-drawn.**
   `corridor = ocean pixels whose mean cur_speed (channel 0, train months
   only, non-holdout stats) is ≥ the 75th percentile of that mean over
   window ocean` → binary-dilate 2 cells → union with the RAPID section row
   (`temporal.rapid_section`). That picks up the Gulf of Mexico loop
   current, Gulf Stream, NAC into the Nordic seas, and the return
   boundary — "the pixels that contribute to the AMOC current" by the
   data's own definition. Report skill over (a) the corridor, (b) the full
   window; the corridor is the headline.
7. **Baseline = the published e017 heads rolled by the NEW evaluator.** Not
   re-trained, not #217's numbers copied. The new evaluator must first
   REPRODUCE #217's truefit numbers with a stencil-1 head (validation gate,
   §6.5); after that gate passes, all arms are scored by the same code.

## 4 · Implementation part A — `ml/temporal.py`

### 4.1 Stencil table

Add near `rapid_section` (module level, importable by the evaluator):

```python
def build_stencil(ocean, ys, xs, stencil):
    """NBR [P, S] int64 indices into the P ordering, -1 = missing
    (land or outside the window; NO lon wrap — regional window).
    Slot 0 is always the centre: NBR[:, 0] == arange(P)."""
    H, W = ocean.shape
    lin = np.full((H, W), -1, np.int64)
    lin[ys, xs] = np.arange(len(ys))
    offs = STENCILS[stencil]
    NBR = np.full((len(ys), len(offs)), -1, np.int64)
    for k, (dy, dx) in enumerate(offs):
        yy, xx = ys + dy, xs + dx
        ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        NBR[ok, k] = lin[yy[ok], xx[ok]]
    assert (NBR[:, 0] == np.arange(len(ys))).all()
    return NBR
```

### 4.2 Model

`TemporalTransformer.__init__` (line ~62) gains `stencil=1`:

```python
self.stencil = stencil
if stencil == 1:
    self.inp = nn.Linear(d_z + 2, d_model)          # EXACT legacy shapes:
    self.static = nn.Linear(d_z + 2, d_model)       # old ckpts load strict
else:
    self.inp = nn.Linear(stencil * d_z + 2, d_model)
    self.static = nn.Linear(d_z + 2 + stencil, d_model)
```

`forward` is unchanged — it already just concatenates and projects; the
caller supplies `z_seq [B, K, S*d_z]` and `static_ctx [B, d_z+2+S]` (or the
legacy shapes for stencil 1). **Back-compat is non-negotiable:** stencil 1
must rebuild every published head with `load_state_dict(strict=True)`. The
model-rebuild sites (temporal.py ~903, rollout.py ~575, and the new
evaluator) read `stencil` from checkpoint args with `.get("stencil", 1)` —
same pattern as `codec_from_ckpt` and the `pos.weight` k_max lesson
(rollout.py ~570: take table sizes from the FILE, never a convention).

### 4.3 The gather helper (replaces five hand-rolled gathers)

```python
def gather_stencil(Zt, Mt, base, p, NBR, K):
    """base [n] window-start month indices, p [n] centre pixels.
    Returns zseq [n, K, S*d_z] float32, mseq [n, K, 2], and (for targets)
    the centre column is NBR[p, 0] == p so ztgt gathers stay as they are.
    Missing neighbours are zero-filled."""
    nbr = NBR[p]                                   # [n, S]
    miss = nbr < 0
    safe = np.where(miss, 0, nbr)
    cols = []
    for j in range(K):
        zj = Zt[base + j]                          # advanced idx -> [n, ...]
        # Zt is a [T, P, d_z] memmap; Zt[base+j] with base a vector gives
        # [n, P? NO] — use Zt[(base+j)[:, None], safe] -> [n, S, d_z]
        zj = Zt[(base + j)[:, None], safe].astype(np.float32)
        zj[miss] = 0.0
        cols.append(zj.reshape(len(p), -1))        # [n, S*d_z]
    zseq = torch.from_numpy(np.stack(cols, 1))     # [n, K, S*d_z]
    mseq = torch.stack([Mt[base + j] for j in range(K)], 1)
    return zseq, mseq
```

(For stencil 1 this reduces to the current gather bit-for-bit; verify in
the unit test, §5.) Call sites to convert — ALL of them, grep for
`Zt[base` and `Zt[_mb` and `Zt[b_` after editing to confirm none remain:
train batch (~1067–1069), unroll futures (~1080 — but unroll is refused for
stencil>1, so it may keep the centre-only gather guarded by the refusal),
monitor (~1151), light probe (~1262), eval (~1336), raw eval (~1388).
Targets (`ztgt`, `ztrue`, `zlast`, `zfut`) remain CENTRE-only gathers —
the model predicts the centre.

`static_ctx` construction (~line 808 region, where it is built today):
append `obs = (NBR >= 0).astype(np.float32)` columns for stencil > 1.

### 4.4 CLI and refusals

- `--stencil {1,9,13}` (default 1, `choices`), stored in the checkpoint via
  the existing `vars(a)` save — no extra plumbing.
- Refuse: `stencil > 1 and unroll > 1` (SystemExit, message says why);
  `stencil > 1 and direct` heads likewise (same feed-back problem).
- Step-time note for sizing: the gather does S× the memmap reads. Measured
  today: 12.4 ms of a 725 ms step at 1.8M params; at 32M/GPU the step is
  ~70 ms, so a 9× gather (~110 ms) becomes the bottleneck → expect
  **~180–250 ms/step, i.e. 60k steps ≈ 3–4 h** per arm on a 4090. If it
  measures worse than 400 ms/step, move the gather to a pinned-memory
  prefetch thread — but MEASURE first (ml/CLAUDE.md §4.8: toy before GPU).

## 5 · Implementation part B — tests (sandbox, CPU, minutes)

New file `tests/test_e022_stencil.py`, following `tests/test_recw_knobs.py`
style (toy npz fixture must include `rapid` and `norm` members):

1. **Zero-weight equivalence (EXACT, the §4.9 invariant).** Build a
   stencil-9 model; copy a trained stencil-1 model's weights into it, with
   `inp.weight` columns for neighbour slots set to ZERO and centre-slot
   columns copied in slot order; static likewise (obs-flag columns zero).
   Assert forward outputs are `torch.allclose(..., atol=1e-6)` on random
   input. This proves the layout mapping is what §3.1 says it is.
2. **Back-compat**: train stencil-1 for 30 toy steps with the NEW code, save,
   rebuild via the checkpoint args, `strict=True`; also load one REAL
   published head spec (skip if offline).
3. **Stencil-9 toy run learns**: 100 steps on a toy field with planted
   advection (roll the field one cell east per step + noise): the stencil-9
   val MSE must beat stencil-1's on the same toy by construction — this is
   the one toy where coupling provably helps, and it exercises the full
   gather + model path end to end.
4. **build_stencil edges**: corner pixel has the right -1 pattern; centre
   column is identity; NO x-wrap (pixel at x=0 has -1 west neighbours).
5. **Refusals fire**: stencil 9 + unroll 4 exits with the message.

## 6 · Implementation part C — `ml/rollout_spatial.py` (new)

GPU-native from day one (the rollout.py CPU head-loop burn cost ~$1.9
across #211/#217 — do not reproduce it). Structure copies
`ml/project_amoc.py` (data loading, Z verify, static encode, truefit ridge
`fit_ridge`, month features) — read that file first; it already solves
everything except the full-window roll.

1. **Inputs**: `--x ml/cache/family3_X.npy --npz-small
   ml/cache/f3_dec_small.npz --z <Zcache> --ckpt <codec> --heads <paths...>
   --out ml/runs/actions/rollout_spatial.json [--horizon 12] [--starts
   heldout] [--long-start 2004-12] [--pixels-gate 600]`.
2. **Static context for ALL P pixels**: encode static channels for the full
   window. project_amoc.py does it for a 3-row slab; here it is the whole
   grid — chunk the codec.encode calls (8192 pixels/chunk, GPU, ~1 min).
   The anomaly stats come from `recon_eval.stream_stats` (cached
   `ml/cache/std_stats.npz`); gather 3×3 codec patches with
   `model.gather_px` over the full X memmap per chunk of rows.
3. **The roll**: state = deque window `Zwin [P, K, d_z]` f32 on CPU (500 MB),
   built from the true Z cache at the start month. Per step: for each pixel
   chunk (8192), assemble `[chunk, K, S*d_z]` via NBR (on CPU, numpy),
   forward on GPU, collect ẑ [P, d_z]; append to window, drop oldest.
   Missing-neighbour handling identical to training (§4.3). With S=13,
   memory per chunk ≈ 8192×24×832×4 ≈ 654 MB — use chunk 4096 if VRAM
   complains.
4. **Scoring** (per head):
   - h=1..12 field skill from held-out-year starts (3 years × 12 offsets =
     36 starts): decode ẑ through `codec.query` (GPU) at each horizon and
     compute `msss_damped`, `acc`, `amp_ratio` exactly as rollout.py's
     chan_skill does (copy the damped-persistence AR1 construction verbatim,
     train months only) — over (a) corridor mask, (b) full window, (c) the
     `--pixels-gate 600` subset with `default_rng(0)` (for the gate below).
   - AMOC truefit at h bands: pool rolled z over the RAPID section, apply
     the ridge fit on TRUE train-month embeddings (fit_ridge from
     project_amoc.py), score in the same h1–3/h4–6/h7–12 bands as
     rollout.py so the numbers are directly comparable.
   - The 240-month hindcast from `--long-start` + a future roll, median
     only (no ensemble here — E-021b covers ensembles): report r on
     trained months vs held-out years, and amp ratio at the 18-month band.
5. **VALIDATION GATE, fatal:** run first with `e017_u1_s0` (stencil 1).
   Its truefit band numbers on the gate subset must match #217's
   (`probes-217.json` → `rollout_eval.json.heads.u1_s0.amoc_bands`:
   h1-3 0.470, h4-6 0.375, h7-12 0.492) within ±0.01, and AUC on the gate
   subset within ±0.01 of 0.643. If they do not match, THE EVALUATOR IS
   WRONG — stop, fix, do not score any spatial head past a failed gate.
   (Differences to hunt first: damped-persistence fit months, deseason
   clim source, section pooling, msss weighting by observed cells.)
6. **Output**: one JSON `rollout_spatial.json` — heads keyed by label
   (`s9_s0` etc. read from ckpt args: `f"s{stencil}_s{seed}"`), each with
   `{gate: {...}, corridor: {...}, window: {...}, amoc_bands: {...},
   long: {...}}` plus a `corridor_def` block recording the percentile, the
   dilation, and the corridor pixel count (no silent caps — record what was
   scored). Add `"rollout_spatial.json"` to `WANT` in
   `scripts/archive_probes.py` so it lands in `probes-<n>.json`.

## 7 · Implementation part D — workflow wiring

Two additions to `.github/workflows/ml-train.yml`. **The Probes `run:`
block has a hard 21,000-character dispatch-time ceiling and sits at 20,023**
("Exceeded max expression length" 422s EVERY dispatch of the workflow —
measured 2026-08-13, see dectrain_run.sh's header). Therefore:

1. **Stencil knob** rides the existing `window` comma format: extend the
   UNROLL/SEED parse (~line 875–895) with ~4 lines:
   ```bash
   STENCIL=1
   case "${{ inputs.window }}" in
     *stencil:*) STENCIL="${{ inputs.window }}"; STENCIL="${STENCIL##*stencil:}"; STENCIL="${STENCIL%%,*}";;
   esac
   ```
   and pass `--stencil "$STENCIL"` where temporal.py is invoked. After the
   edit, MEASURE the block:
   ```bash
   python3 -c "import yaml; d=yaml.safe_load(open('.github/workflows/ml-train.yml'));
   [print(len(s['run'])) for j in d['jobs'].values() for s in j['steps']
    if s.get('name')=='Probes (K-sweep + stage 2)']"
   ```
   If it exceeds ~20,800, first move an existing comment out of the block.
   Count the inputs too — must stay exactly 25.
2. **Eval token** `sroll:<tag,tag,...>` → `bash scripts/sroll_run.sh
   "${{ inputs.window }}"` + `exit 0`, a case branch in the same block
   (model it on the `project:*` branch, 6 lines). `scripts/sroll_run.sh`
   copies `scripts/project_run.sh` nearly verbatim (extract X, find Z, pull
   heads with `curl -fsSL` — boxes have no gh/node), calling
   `ml/rollout_spatial.py` and asserting the output file parses with a
   non-empty gate block before exit 0.

## 8 · Dispatch plan (run this order, verify between steps)

All dispatches use the #211/#219 provenance template verbatim (fetch
`probes-219.json → files.provenance.json.inputs` for the exact 25 keys)
with only `doc`, `window`, `runner`, `job_timeout` changed. Both boxes are
STOPPED as of 2026-08-13 12:30Z — `node scripts/gpu_box.mjs start 47483091`
(gpu-box-40623952) / `47487801` (gpu-box-42005419); runner reconnects in
~2 min; check `actions/runners` shows it online before dispatching.

- **R0 (sandbox, free):** `python -m pytest tests/test_e022_stencil.py` —
  all green before anything is dispatched.
- **R1 (one box, ~30 min):** ONE short spatial run as a live smoke:
  `window: "stencil:9,seed:0"`, `temporal_steps: 6000`, job_timeout 200.
  Verify in the first minutes (ml/CLAUDE.md §2): step time (expect
  180–400 ms), loss falling, no shape errors; then let it finish and check
  `temporal.json` exists in `probes-<n>` (a green run without it = dead
  trainer, §7 lore). Throw the head away — 6k steps is not an arm.
- **R2 (both boxes, ~10 h wall, ~$6):** six training runs, 60k steps each —
  `stencil:9,seed:{0,1,2}` pinned to gpu-box-40623952 (queue 3), and
  `stencil:13,seed:{0,1,2}` pinned to gpu-box-42005419 (queue 3).
  job_timeout 600. Write the **E-022 EXPERIMENTS.md entry at THIS moment**,
  hypothesis + falsifier + this table, and push it before the dispatches.
- **R3 (publish):** `node scripts/publish_heads.mjs --runs
  <n0>:u1s0,<n1>:u1s1,<n2>:u1s2 --prefix e022s9` and likewise
  `--prefix e022s13`. The claim regex is `^u(\d+)s(\d+)$` — the stencil
  lives in the PREFIX, the claim stays `u1sN` (all these runs are U=1).
  Published tags: `e022s9_u1_s0` … `e022s13_u1_s2`.
- **R4 (eval, one box, ~1–2 h):** `window:
  "sroll:e017_u1_s0,e017_u1_s1,e017_u1_s2,e022s9_u1_s0,...,e022s13_u1_s2"`
  (all nine heads; the e017 trio is the baseline arm AND the gate).
  job_timeout 400.
- **R5 (harvest):** pull `probes-<n>.json` → `rollout_spatial.json`; check
  the GATE block FIRST; then build the table (§9). Stop both boxes
  (`gpu_box.mjs stop`), verify via the jobs API that nothing is queued.

If a run fails: read the log before re-dispatching (§7 failure signatures);
a green run with no temporal.json is a dead trainer; queued-against-idle
> 5 min = cancel and re-dispatch; NEVER stop a box mid-job.

## 9 · Reporting

The RESULT table, three seeds per cell, mean [min–max]:

| metric | stencil 1 (e017) | 3×3 (e022s9) | 13-pt (e022s13) |
|---|---|---|---|
| AUC(msss_damped) corridor | | | |
| AUC(msss_damped) full window | | | |
| amp ratio at h=12 | | | |
| AMOC truefit h1–3 / h4–6 / h7–12 | | | |
| long-hindcast r trained / held-out | | | |
| nowcast k-fold (from temporal.json) | | | |

Decision rules, pre-registered: primary = corridor AUC vs the stencil-1
band (§1 falsifier). If 13-pt > 3×3 > 1 monotonically outside seed spread,
reach matters → a wider-operator follow-up (E-023) is justified. If both
spatial arms ≈ stencil 1, close the local-coupling axis at monthly cadence
and record the physics caveat as the likely reason. If spatial arms are
WORSE (the E-020 pattern), say so plainly — a bigger input that hurts is
evidence about optimisation, and the zero-weight-equivalence test (§5.1)
proves the model could always have ignored the neighbours, so a regression
means training dynamics, not capacity.

Also report COST (dispatches, GPU-hours, failures) per ml/CLAUDE.md §3.

## 10 · Budget

~30 min sandbox tests (free) + R1 ~$0.15 + R2 ~$5.50 + R4 ~$0.50 ≈ **$6.5,
~11 h wall** with both boxes. Idle-tail discipline: stop boxes at R5.

## 11 · Pitfalls that WILL bite (each has bitten before)

- Probes `run:` block ceiling 21,000 chars (dectrain incident) — measure
  after every workflow edit; 25 inputs exactly.
- Boxes: no `gh`, no `node`; `curl -fsSL` for release assets; `curl -T`
  for uploads; `bash -e` kills best-effort steps — guard at the CALLER.
- Brace every `$VAR` followed by `_`/letter/digit: `"${TAG}__temporal.pt"`.
- Green run ≠ trained run: the archive's file list is the truth.
- `pos.weight` k_max and every architecture knob come from the CHECKPOINT,
  never a convention (two failures each way already).
- The sandbox container restarts ~every 2 h: commit + `node
  scripts/git_api_push.mjs --token-file /home/claude/.gh_pat --branch main`
  after every edit that matters; the git proxy blocks plain push.
- Vast list endpoint is v1 (`/api/v1/instances/`); v0 silently 410s.
- Runner names ≠ instance IDs; map via the onstart script before stopping.
- EXPERIMENTS.md entry at dispatch. E-021 has the recorded violation;
  E-022 should not be the second.

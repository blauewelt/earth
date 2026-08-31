# Session handover · 2026-08-31

**Written for a session starting cold — possibly on a different Claude
subscription, with none of this project's chat history and possibly none of
its project docs.** Everything load-bearing is either in this file or named by
a path inside the repo.

The programme just went through the largest correction in its history. A new
session must understand that correction before it reads any older number,
because most of the older numbers do not mean what they say.

---

## 0. Read these, in this order, before doing anything

1. `ml/OVERVIEW.md` — the standing map. Its top block is **THE PROTOCOL
   RESET**; that block supersedes everything below it in the same file.
2. `ml/plans/PROTOCOL_RESET.md` — what may count as a result after
   2026-08-29, and the four memorisation signatures that forced it.
3. `ml/plans/DATA_LADDER.md` — what data to import next and in what order.
4. `ml/EXPERIMENTS.md#e-062` — the first honest rolled number, plus the
   metric correction.
5. `ml/CLAUDE.md` — the working principles. §0.2 (a backup is only real if
   the restore works), §0.3 (preconditions fire before the expensive part),
   §1 (write the experiment entry AT dispatch, not after).

All of these render at
`https://blauewelt.github.io/earth/docs.html?f=<path>`.

---

## 1. What the programme is

Predict North Atlantic ocean state forward, with **AMOC transport at 26.5°N
(RAPID)** as the headline read-out.

Two stages:

- **Stage 1, the codec.** A masked autoencoder ("PixelMAE") compresses each
  ocean pixel's 40 physical channels into a 32-dimensional embedding `z`.
  Trained self-supervised, then **frozen**.
- **Stage 2, the head.** A causal transformer over a window of K=144
  embeddings at one pixel plus a 145-point spatial stencil, predicting
  `z_{t+1}`. This is what "forecasting" means here.
- **Read-out.** Linear/ridge probes from the embeddings to RAPID transport;
  and a spatial roll (`ml/rollout_spatial.py`) that advances every pixel
  forward and scores the resulting fields.

Tensors: `family3_na025` (monthly, T=516), `family4_na025_pentad_r2`
(5-day, T=3,142, 40 channels, 86,698 ocean pixels — **the current working
tensor**), `family5_na025_daily` (T=15,706, built once, 165.6 GB, stage-1
only), `family6_na025_cmip6` (new, see §7).

---

## 2. The correction — what happened on 2026-08-28 to 08-30

### 2.1 The pool bug

Stage 2's loss is **dense** over the window: every one of the 144 frames is
supervised against the embedding one bin after its own time. The training
pool did not know that — it admitted a window if and only if its **final**
scored bin was not held out. So windows straddling a holdout year were
admitted and **teacher-forced that year's transitions into the weights**.

Measured on the real axis: **21,018 of 400,176 scored frame-targets were
held-out bins**. Fixed in commit `c25f6ff` by `--holdout-scope`, which now
takes three values:

| scope | rule | cost in supervision |
|---|---|---|
| `endpoint_contaminated` | legacy; only the final scored bin must be clean | — (leaky) |
| `target` | held-out targets masked from the loss; held-out bins may still be context | −5.25 % |
| `window` | nothing the forward pass touches may be held out | −13.03 % |

`window` is the default. The pool self-certifies by brute force before
training and `sys.exit`s on a violation.

### 2.2 The four memorisation signatures

Each from a published artefact, not an argument:

1. **The pool bug itself** (above).
2. **Skill that does not decay with lead.** #510 re-rolled the old head:
   field anomaly correlation **0.985 at 5 days and 0.973 at 365 days**, flat
   across the whole year. A year-ahead prediction as good as a five-day one
   is a replayed trajectory.
3. **No spatial generalisation.** #513: corridor **0.838 on trained
   longitudes against 0.176 on held-out ones** (gate head 0.804 vs 0.058).
   On ocean it has not seen, the head removes about 6 % of the anomaly
   variance. Every headline aggregate ever quoted is a ~24 %-untrained blend.
4. **Confidence exactly where it memorised.** #513's dispersion: eight
   independently-noised trajectories stay pinned together over years the
   model has seen (0.1060 → 0.1015 over 36 months) and fan out over years it
   has not (0.1974 → 0.4197). A memorisation test with no labels in it.

### 2.3 The metric correction

**What this programme calls "corridor AUC" is not an AUC.** It is
`horizon_auc` = mean `msss_clim` = `1 − MSE_model / MSE_climatology`, on
fields that are already anomalies — so "climatology" means predicting zero.
**1.0 perfect, 0.0 no better than predicting nothing, negative = error
exceeds the anomaly variance.** Negative is *not* "below chance" and does not
mean a ranking inverts. `ml/rollout_spatial.py:117` has said so since E-017;
nobody read it.

Every "0.9xx AUC" in the older text is a `msss_clim` from a contaminated
head. Treat all of them as retired.

### 2.4 The first honest number — E-062-R0 (#516)

Same battery, one variable changed (the training pool):

| | h=1 (5 d) | h=73 (365 d) | mean |
|---|---|---|---|
| #516 · clean pool | **0.606** | **−0.031** | 0.105 |
| #510 · contaminated | 0.985 | 0.973 | 0.946 |

**The clean head passes the lead-decay falsifier its contaminated twin
fails.** It decays like a forecast. Corridor `horizon_auc` +0.888 → **−0.439**.

The negative score is a **calibration** failure and the arithmetic closes:
mean `acc` 0.105 against mean `amp_ratio` 0.780 — the head keeps emitting
anomalies at 78 % amplitude long after it stops knowing their sign. The
identity `msss = 1 − (1 + a² − 2a·ACC)` reproduces all 73 leads to a mean
absolute error of 0.0135. **Amplitude-calibrated, the same states would score
+0.019 instead of −0.439.** The head also beats persistence at every lead but
the last (mean `msss_pers` +0.204).

Where residual skill lives: **SST** beats climatology out to 90 days
(+0.069); **SSH** is best at 5 days (+0.717); and only 8 of 40 channels are
scored at all — the 32 Argo `rg_*` channels are null at every lead.

### 2.5 The capacity result, and my failed prediction

E-060 ran a width ladder under the clean pool: 7,597,856 / 40,388,128 /
206,658,592 / 399,947,552 parameters, everything else identical.

| params | best held-out ratio | at step | ratio @20k | RAPID probe |
|---|---|---|---|---|
| 7.6M | **0.60951** | 1,200 | 0.69218 | 0.598–0.611, flat |
| 40.4M | 0.62121 | 1,200 | 0.63628 | drifts to 0.589 |
| 206.7M | 0.61049 | 2,000 | 0.64106 | **0.616 → 0.515** |
| 400M | 0.57793 | 800 | (only 2k steps run) | — |

**I pre-registered that 7.6M would land within 0.02 of 206.7M at step 20,000
and it did not — it is 0.051 worse.** That is recorded as a failed
prediction in `ml/EXPERIMENTS.md#e-060-read`, and a new session should not
find a softened version of it anywhere.

What survives: every arm reaches its best held-out loss **inside 2,000 steps
of a 200,000-step run** and worsens for the remaining 198,000; the best level
sits in 0.58–0.62 across a **53× parameter span**; and the degradation is
width-dependent — the large arm collapses its RAPID probe while the small one
holds. **Small models are right for cost, not for cause.** The constraint is
2,417 end-bins.

### 2.6 The frozen protocol (decided by Chris, 2026-08-30)

- **Terminal holdout: train ≤ 2020, test 2021–2024, no gap.** (The tensor
  ends 2024-12-31; 2025 does not exist.)
- `_trainlon` / `_holdlon` reported separately as the headline, **never the
  blend**.
- **Lead-decay as a falsifier** on every rolled result.
- A **null ladder** on every number, including a nearest-analogue retrieval
  baseline.
- **Early-stop at the held-out minimum.**
- **Rolled skill is the verdict, never a probe.**
- Small tier (7.6M) is the default; 206M+ needs a specific argument.

---

## 3. Open questions — the honest list

These are open in the sense that the programme cannot currently answer them,
not in the sense that nobody has an opinion.

**Q1 · Is there forecast skill at all beyond about ten days?**
R0 is the only clean rolled number and it reads: field `acc` 0.606 at 5 days
falling to −0.031 at a year; transport bands 0.107 / −0.242 / 0.163, all
inside noise of zero. That is consistent with "a few days of real skill and
nothing after", and also with "the head is badly calibrated and the skill is
hiding". **n = 1**, at a cadence with no replicate pair, on the 200k memorised
end state. This is the central question and it is unresolved.

**Q2 · Is the amplitude calibration a real +0.46?**
The arithmetic says calibrating `a = ACC` turns −0.439 into +0.019 on states
already computed. It is a decoding change, not a retrain. Untested. If it
works it is the cheapest large gain available; if it does not, the identity
above is wrong somewhere and that is worth knowing.

**Q3 · Does the step-2,000 checkpoint roll better than the 200k one?**
Every arm peaks at ~2,000 steps. Every rolled number so far comes from a
200k end state, i.e. from past the held-out minimum. **Rolling the early
checkpoint is the cheapest untried thing in the programme** and no one has
done it.

**Q4 · Does the head generalise spatially at all?**
#513 says essentially not (0.176 held-out longitude). R0 could not measure it
because the pentad r2 tensor has no longitude hole, so `_trainlon` equals the
parent. A longitude-holdout retrain under the frozen protocol is required
before any spatial claim.

**Q5 · Is there enough statistical power to decide anything?**
Roughly **9 effective starts**. RAPID is ~20 years; AMOC anomalies
decorrelate over months to years, so the independent-sample count for the
headline target is order 10², against models of 10⁷–10⁸ parameters. It is
entirely possible that several of the questions above **cannot be resolved
with this data at all**, and establishing that would itself be a result.

**Q6 · Is the target predictable, by anything?**
No upper bound has ever been established. The protocol calls for a
**nearest-analogue retrieval baseline** — find the most similar past state,
copy what happened next — and it has not been run. If retrieval matches the
transformer, the transformer is an expensive lookup table. This should be
run before any further architecture work.

**Q7 · Does pretraining on climate-model output transfer, or does it just
distil the model?**
Chris's own question, and it is the right one. See §7.

**Q8 · Are 32 of the 40 channels dead weight?**
R0's per-channel decomposition scores them null at every lead. They are
1° native, monthly, start in 2004, are live one pentad in six, and cost
**27.2 GB — 80 % of the tensor — to carry about 0.28 GB of information**
(~97× redundancy). Dropping or thinning them is nearly free and frees more
storage than every proposed import combined.

---

## 4. The research challenges ahead

**4.1 The wall is 43 years, and everything else is downstream of it.**
The training pool's headline `209,549,066` windows is exactly
`2,417 end-bins × 86,698 pixels`. There are 2,417 *distinct temporal
patterns*, and the pixels are correlated views of them. No architecture
change alters that number. This is the single fact a new session should keep
in front of it.

**4.2 Memorisation is the default behaviour, not an accident.**
With 206M parameters and 2,417 windows there are ~85,000 parameters per
distinct pattern. Fixing the pool removed the *reward* for memorising the
evaluation years; it did not remove the *incentive* to memorise the training
years. Early stopping, small models and heavy regularisation are palliative.

**4.3 The evaluation stack has been wrong twice.**
Once structurally (the pool), once semantically (calling `msss_clim` an
AUC). Both survived months of use. A new session should assume there are
more, and should treat "verify the metric" as ordinary work rather than
paranoia.

**4.4 The two-stage architecture has never been tested end to end under a
clean protocol.**
The codec is frozen and everything downstream inherits whatever it discards.
Whether the 32-dimensional bottleneck is the limiting factor is unknown; the
joint fine-tune path exists in the workflow and has not been exercised since
the reset.

**4.5 Calibration versus skill.**
R0 shows a head that is confidently wrong: 78 % amplitude, 10 % correlation.
Whether the programme wants a well-calibrated damped forecast or a sharp
uncertain one is a design decision nobody has made. The FGN thread (E-057,
`ml/plans/E057_fgn_head.md`) is an attempt to sample rather than emit the
conditional mean; it is currently **retired from the ranking** but the
question it asks is live.

**4.6 Cost and cadence of experimentation.**
One 200k-step run is ~16.5 h on a TPU v5e-4. Spot quota is **4 cores per
zone** and three of the five reachable zones have no capacity, so waves run
sequentially. This shapes what is answerable per week more than any
scientific consideration.

---

## 5. Infrastructure a new session will trip over

- **Pushing to `blauewelt/earth` fails through the sandbox git proxy (403).**
  Use `node scripts/git_api_push.mjs --token-file /home/claude/.gh_pat
  --repo blauewelt/earth --branch main --range origin/main..HEAD`. Other
  sessions push to `main` too — rebase first, never force.
- **Credentials live in files, never on command lines.** The permission
  classifier blocks tokens in argv, correctly.
- **`ml-train.yml` is at the 25-input ceiling.** A 26th input makes the whole
  workflow unparseable. New capabilities go in as *values* on an existing
  input (that is how `family6_na025_cmip6` was added).
- **`mint_token()` in `scripts/tpu_box.py` returns a TUPLE — take `[0]`.**
- **No shell reaches a running Vast box.** Work on it goes through a
  dispatched Actions job.
- **Vast boxes exit after provisioning** more often than you expect; check
  `node scripts/gpu_box.mjs list` and `start <id>` before blaming a queue.
- **Disk is the most common cause of death.** Two runs (#508, #509) died on
  a full 100 GB disk, one of them before any cleanup step could run.
- **TPU node names are the resume key** — relaunching under the same node
  name continues from the shipped checkpoint.

### Credentials, and the risk if the project does not travel

These live in **project docs**, not the repo. On a new subscription without
this project, **they will have to be re-supplied**:

| what | project doc | sandbox file |
|---|---|---|
| GitHub PAT | `claude/github-access.md` | `/home/claude/.gh_pat` |
| Vast.ai | `claude/vast-access.md` | `/home/claude/.vast_key` |
| GCP service account (TPUs + `gs://earth-tpu-staging`) | `ml/plans/TPU_ACCESS.md` | `/home/claude/.gcp/sa.json` |
| Hugging Face | `claude/huggingface-access.md` | `/home/claude/.hf_token` |
| Copernicus Marine | `claude/copernicus-marine-access.md` | env only, by policy |

**Two credential facts worth carrying forward.** The repo has **no Actions
secrets at all**, and the PAT returns **403** trying to write one — so any
job needing a token in CI is blocked until a human adds it in Settings →
Secrets. And the "one blocker" recorded in `PROTOCOL_RESET.md §6` — that a
session could not obtain the GCS read credential to mirror the 7.6M head —
**was a property of that session, not of the project**: the key is at
`/home/claude/.gcp/sa.json` and works. See §8.

---

## 6. What data we are considering adding

`ml/plans/DATA_LADDER.md` is the full document. Its central framing, which is
the part that matters:

| class | what it adds | does it move the wall? |
|---|---|---|
| **new temporal samples** | more distinct end-bins | **yes — this is the constraint** |
| new spatial detail | more pixels inside the same 43 years | **no** |
| new physical variables | channels the tensor lacks | orthogonal, and cheap |

**Only the first class touches the measured bottleneck.** Going to 1/12°
would make the pool `2,417 × 780,282` — 1.9 billion windows from the *same
2,417 bins*, at nine times the disk. Correctly declined.

The ranked list, with my assessment of each:

1. **GREP 3-D at 0.25°, 4 variables × 8 levels (+20 GB).** Same product,
   credentials, grid and code path already in use. Subsurface velocity and
   abyssal levels the tensor has never had, 11 extra years, 6× the live
   subsurface bins. *Best physics-per-risk on the list — but it does not move
   the wall.*
2. **The statics and RAPID's unused files (~3 MB).** Bathymetry, slope, mask,
   Coriolis `f`, `β`, distance-to-coast; `moc_vertical` as a depth-resolved
   label. *Highest information per byte in the whole document. Afternoons,
   not projects.*
3. **A CDS account, then ERA5 forcing (+5 GB net).** **Buoyancy forcing is
   entirely absent from the tensor today**, and it is the mechanism the
   subpolar precursor is supposed to run through. *This is missing physics,
   not more data — arguably it should be item 1.*
4. **DUACS + OSTIA at 0.25° (+4.7 GB).** The observed surface overturning
   signature and the missing sea-ice channel. Existing credentials.
5. **The backward extension to 1958 (~3 GB).** EN4.2.2 (1900→, 1°, 42
   levels) or IAPv4 (1940→, temperature only) as a 1° T/S-only tensor.
   **This is the only candidate on the entire list that attacks the measured
   constraint**, and it is cheap enough to run as a measurement first: build
   it, train a small-tier head, and ask whether the step-2,000 wall moves.
   *Whichever way it answers, it is the most valuable item here.*
6. **The 2025–2026 continuation (+4.5 % of end-bins).** Nearly free, and it
   is what would let the terminal holdout run to 2026 instead of 2024.

**Skipped, and rightly:** MUR at 1 km · basin-wide GLORYS12 at 1/12° · ARMOR3D
at native 0.125° · SMOS/SMAP · ocean colour.

**Before any import:** move the 32 `rg_*` channels to a 1° sidecar and
upsample in the loader. That frees **~27 GB — more than items 1, 3, 4 and 6
cost combined** — and it is not a data import at all.

### Does adding data make sense? An honest answer

**Partly, and less than one would hope.**

The wall is not a shortage of pixels, channels or bytes. It is a shortage of
**independent realisations of the thing being predicted**. There are 43 years
of ocean reanalysis and about 20 years of RAPID, and the North Atlantic
overturning varies on interannual-to-decadal timescales. No import creates
more North Atlantic.

So the honest ranking of *purposes*:

- **Items 1–4 improve the inputs.** They give the codec a more faithful
  pixel — subsurface structure, real buoyancy forcing, the observed sea
  level. That is worth doing on its own merits, and it may raise the ceiling
  on what a well-posed forecast could achieve. It will not move the
  step-2,000 wall, and nobody should expect it to.
- **Item 5 is the only one aimed at the wall**, and it should be run as a
  measurement before it is run as a commitment.
- **Item 8 in §3 — deleting channels — may be worth more than any import.**
  80 % of the tensor's bytes score null at every lead.

---

## 7. E-061 — the CMIP6 corpus, and why it may not be the answer

**Built.** `ml/build_family6.py` streams CMIP6 `piControl` from the public
Google Cloud zarr store (anonymous HTTPS) and regrids it onto family 3's
exact axes: HadGEM3-GC31-MM (500 model-years) + CNRM-CM6-1-HR (300), both
NEMO eORCA025 at the same nominal 25 km as the target. Variables `mlotst`,
`zos`, `tos` → `log_mld`, `ssh`, `sst`. **All scalars, deliberately** —
`uo`/`vo` on a tripolar C-grid carry a local rotation angle and regridding
them unrotated produces a systematically wrong subpolar gyre inside our own
window.

Verified: 1-NN map on 3-D unit vectors, **max nearest-neighbour 15.02 km**
(refuses over 40); against GREP's January climatology **spatial correlation
0.9927**, mask disagreement 0.46 % and those cells are the Great Lakes, which
HadGEM3's ocean carries and a reanalysis does not.

Stored **sparse**: `[9600, 281, 481, 3]` float16 = **7.79 GB**, where the
naive 40-channel dense array is 103.8 GB of which 92.5 % is the literal
absence of Argo and NCEP over a model run that never had them.
`channel_index = [1, 2, 39]` and `expand_to_full()` scatter it back.

**Status: the corpus exists on Vast box `gpu-box-35586926` and is
unpublished.** #517 built it in 80 minutes, then the job was marked failed by
the *provenance* step reading `d["months"]` — family 6 deliberately writes no
`months` key, so a model-year of a free-running control run cannot be
mistaken for a calendar year. Fixed in `1c2d205`. Publishing it needs the
`HF_TOKEN` Actions secret, which does not exist.

### Whether it makes sense — the distillation objection

Chris put it directly: *"aren't we just distilling the same model that was
used to create the hindcast?"* Largely, yes.

- **Our existing training data is also model output.** GLORYS12 is NEMO with
  observations assimilated. HadGEM3's and CNRM's oceans are NEMO. This is not
  a step from real data to synthetic data; it is a step from *NEMO nudged
  toward observations* to *NEMO running free*.
- **That weakens the objection**, because the dynamics are the same
  discretised equations the reanalysis obeys — geostrophy, thermal wind,
  Sverdrup balance, the seasonal mixed-layer cycle are shared with reality.
- **It also weakens the value.** If the corpus shares the reanalysis's
  structural biases it teaches no new dynamics, only *more samples of the
  same dynamics*. Which is, to be fair, exactly the diagnosed deficit.
- **Where it could actively hurt:** CMIP6 AMOC strength spans roughly
  10–30 Sv against RAPID's ~17, and AMOC *variability* is among the most
  model-dependent quantities in the archive. **The one thing we are trying to
  predict is the thing these models are least trustworthy about**, and the
  network cannot be told which parts to learn.

**Two better uses of the same corpus, neither yet tried:**

1. **Pretrain the codec, not the forecaster.** Learning what ocean states
   look like spatially is far less sensitive to a model's AMOC bias than
   learning the forward map.
2. **Use GREP instead.** Three observationally-constrained realizations of
   1993–2024 (CGLORS/CMCC, GLORYS2V4/Mercator, ORAS5/ECMWF), already
   downloaded to `/home/claude/hindcast/` and verified on our exact grid. A
   3× multiplier rather than 20×, with far less bias risk. If the deficit is
   sample count, this is the low-bias version of the same test.

**The confound any follow-up must control for, registered before the fact:** a
3-of-40-channel corpus fed to a 40-channel codec is a **distribution shift**
as well as a transfer. The control is to embed the *reanalysis* with the same
37 channels masked, so "pretraining helped" can be distinguished from "the
fine-tune adapted to a channel set it had been pretrained on". Without that
control the result is uninterpretable in either direction.

**Verdict:** worth finishing because it is cheap (~$5 of compute, already
mostly spent) and genuinely falsifiable. Expectations should be low, and it
should not be the programme's main line. Item 5 of §6 — the backward
extension to 1958 — attacks the same constraint with observationally
constrained data and deserves priority.

---

## 8. Immediate next actions

In order of value per unit of effort:

1. **Roll the step-2,000 checkpoint** (Q3). Cheapest untried thing in the
   programme, and every rolled number to date comes from past the held-out
   minimum.
2. **Apply the amplitude calibration** (Q2) to states already computed. A
   decoding change worth ~+0.46 `msss` if the identity holds.
3. **Mirror `temporal_e060a.pt`** (30,425,836 bytes) from
   `gs://earth-tpu-staging/runs/e060a-8m/` to the `model-checkpoints-v1`
   release, so the roll evaluator can read it. **This is recorded as the
   programme's one blocker and it is not blocked** — the GCS key at
   `/home/claude/.gcp/sa.json` reads the bucket fine. The remaining work is
   naming the asset to the evaluator's convention; see
   `scripts/publish_heads.mjs` for how a head is verified against its own
   args before upload, and do not skip that check. `temporal_e060b.pt`
   (161,602,140 bytes) is there too.
4. **Run the nearest-analogue retrieval baseline** (Q6). Until it exists no
   rolled number has a null to be measured against.
5. **Drop or thin the 32 `rg_*` channels** (Q8) — frees 27 GB and R0 says
   they contribute nothing at any lead.
6. **The terminal-holdout retrains at 7.6M** under the frozen protocol.
7. **Add the `HF_TOKEN` Actions secret** (needs a human) and re-run the
   family-6 publish so the corpus stops living on one rented disk.

**Do not** start new architecture work before items 1, 2 and 4. The
programme's history is that evaluation errors outran modelling gains by a
wide margin, and three of those four items are evaluation.

---

## 9. Things I got wrong in this session, recorded so they are not repeated

- I claimed a 365-day roll walks into the following year and leaks that way.
  It does not — `ml/rollout_spatial.py:880` breaks at the year boundary, and
  the artefact's own per-horizon `n` (2,171,138 / 1,447,424 / 723,710) proves
  it. The real leak was training-side, §2.1.
- I pre-registered that 7.6M would land within 0.02 of 206.7M at step 20,000.
  It landed 0.051 away. I reported "the prediction is landing" on the
  strength of an early best before the registered step had been reached.
- I wrote `ml/plans/E059_holdout_window.md` and committed it **empty**,
  twice, without noticing — `docs.html` registers the path, not the size.
  Check `wc -c` on a plan after writing it.
- I guarded the Train and Probes steps against the family-6 tensor and forgot
  the provenance step between them, so a bookkeeping step failed after the
  7.79 GB artefact it describes was already complete.

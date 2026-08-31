# Earth · reboot handover · 2026-08-31 (chat session → Cowork)

**What this file is.** A second-opinion review of `SESSION_HANDOVER_20260831.md`
(written by Opus at the end of the 08-28→08-31 session) plus the full transcript
of that session, and a proposed reboot of the research programme.

**It does not replace the Opus handover.** That document is the factual record —
every measured number below comes from it or from the transcript. This file adds
an outside assessment, four things that document does not contain, and one
decision that is still open.

**Read order for a cold Cowork session**

1. `SESSION_HANDOVER_20260831.md` (the factual record — read it first, in full)
2. this file (assessment + proposed plan)
3. then, in the repo: `ml/OVERVIEW.md` (top block = THE PROTOCOL RESET),
   `ml/plans/PROTOCOL_RESET.md`, `ml/plans/DATA_LADDER.md`,
   `ml/EXPERIMENTS.md#e-062`, `ml/CLAUDE.md`

All repo docs render at `https://blauewelt.github.io/earth/docs.html?f=<path>`.

---

## 1. The two premises, assessed

Chris's framing entering this session was: (a) the programme trained on
contaminated data from the beginning, so nearly all results measure memorisation;
(b) "2,417 distinct temporal windows against 206 million parameters" is silly.

**Both hold. Each needs one sharpening.**

### 1.1 Contamination — correct, but the wreckage is not total

Every rolled headline through E-051 / #510 is retired. The decisive artefact is
the flat skill-vs-lead profile: field anomaly correlation **0.985 at 5 days,
0.973 at 365 days**. A year-ahead prediction as good as a five-day one is a
replayed trajectory, not a forecast.

But E-062-R0 (#516), the first clean rolled number, is not zero:

- it **decays like a forecast** — 0.606 at 5 days → −0.031 at a year;
- it **beats persistence at every lead but the last** (mean `msss_pers` +0.204);
- **SST beats climatology out to 90 days** (+0.069); **SSH** is strongly positive
  at 5 days (+0.717).

So the honest statement is: a few days-to-weeks of real surface skill, acquired
inside the first ~2,000 steps of every run, and then ~99 % of every training
budget spent memorising. Smaller than claimed by an order of magnitude — but not
nothing, and where the residual skill lives is measured, not guessed.

### 1.2 The parameter/sample mismatch — correct, and worse than stated

2,417 is itself an over-count of independent information: consecutive windows
share **143 of 144 frames**, and the 86,698 pixels are correlated views of the
same 2,417 temporal patterns (the pool's headline 209,549,066 is exactly
2,417 × 86,698).

For **field-level** prediction the effective sample count is nonetheless in the
thousands — enough to fit something.

For the **headline AMOC target** it is not. RAPID is ~20 years, AMOC anomalies
decorrelate over months-to-years, giving the handover's **~9 effective starts**.
That number, not 2,417, is what should drive strategy. It is the reason for the
open decision in §5.

### 1.3 One contamination question nobody has asked yet

**Was the stage-1 codec trained on the holdout years?**

It was trained self-supervised on the full record and then frozen. The reset
fixed the stage-2 *pool*; it did not, as far as either document records, revisit
the codec. Per-pixel reconstruction is a much milder leak than teacher-forced
temporal transitions — but the RAPID probes read directly out of `z`, so a
strictly clean terminal-holdout protocol arguably requires a codec trained
≤ 2020 as well.

**Action: check and record a ruling before the terminal holdout is ever spent.**
It is a cheap check (read the codec run's split config) and an expensive mistake
if wrong — the terminal holdout can only be spent once.

---

## 2. What the Opus handover already gets right, and should be kept

Adopt these unchanged. They are the product of the correction and they are sound:

- **`window` holdout scope as the default**, with a pool that self-certifies by
  brute force and `sys.exit`s on violation.
- **The metric correction.** `horizon_auc` is `msss_clim`, not an AUC. 1.0
  perfect, 0.0 = no better than predicting zero, negative = calibration failure,
  **not** "below chance". Every "0.9xx AUC" in older text is retired.
- **Lead-decay as a standing falsifier** on every rolled result.
- **`_trainlon` / `_holdlon` reported separately, never blended.**
- **Early-stop at the held-out minimum** — every arm peaks near step 2,000 of
  200,000 and worsens for the remaining 198,000.
- **Small tier (7.6M) as default**; anything larger needs a specific argument.
  Note the recorded caveat: at the registered step 20,000 the 7.6M arm was
  **0.051 worse** than 206.7M, not within 0.02 as pre-registered. Small models
  are right for cost and for probe stability, not proven right for loss.
- **Rolled skill is the verdict, never a probe.**
- The four immediate actions in §8 of that document, in its order.

---

## 3. What this session adds

Four items not in the Opus handover.

### 3.1 A Linear Inverse Model belongs in the null ladder

The handover calls for a nearest-analogue retrieval baseline (its Q6). Correct,
and it should be extended to three nulls, all cheap, all runnable before any
training dispatch:

| null | what it tests |
|---|---|
| **nearest-analogue retrieval** in `z` | if it matches the head, the head is an expensive lookup table |
| **damped persistence** (per-channel AR(1) decay to climatology) | the classic honest null; R0 beats plain persistence at all but the last lead, so this is a real bar |
| **Linear Inverse Model** on leading EOFs of the anomaly fields (or a linear map on `z`) | ~10³ parameters, well-posed at exactly this sample count, and frequently state of the art for SST at S2S horizons |

The LIM is the important addition. "206M transformer vs 10³-parameter LIM on the
same clean protocol" is the capacity question asked properly — far more
informative than a width ladder spanning 53× of the same architecture. **If the
LIM wins, that is the programme's central result**, and it is a good one.

### 3.2 The statistics machinery is missing, and n≈1 is the current state

The frozen protocol (train ≤ 2020, test 2021–2024, no gap) is right as the
*final, touch-once* test. It is not a development protocol — a single terminal
holdout gives n = 1, which is exactly the epistemic position R0 is in now
(the handover's Q1 says so: "n = 1, at a cadence with no replicate pair").

Add, before any further architecture work:

- **Rolling-origin blocked cross-validation** over multiple held-out years,
  each under `window`-scope exclusion, for development decisions.
- **Block-bootstrap confidence intervals** on every skill number at every lead.
  The transport bands (0.107 / −0.242 / 0.163) are currently reported as "inside
  noise of zero" by argument rather than by an interval.
- **Pre-computed minimum detectable effect** — decide before running what
  difference the data could resolve. Several open questions may be formally
  unresolvable, and establishing that is itself a result (handover Q5).
- **Seed ensembles at the 7.6M tier.** At ~1.7 h per arm, 5 seeds is affordable
  and buys error bars plus a spread-skill diagnostic the programme entirely
  lacks. Note the existing dispersion result (#513) was a *memorisation* test;
  this is different — it is a calibration diagnostic.

### 3.3 `paper.tex` must be quarantined now

Every corridor number in the paper is a retired `msss_clim` from a contaminated
head. Until §4's work lands, the results sections should carry an explicit
invalidation note so no one — including a future session — reads them as
current. Separately, the paper still has **no Acknowledgements and no Data
Availability section**, which is owed for the sources already in Table 1
(CMEMS/GLORYS/GREP licensing, Argo, NCEP R1, OISST, GPCP), independent of
anything CMIP6.

Depending on how §4 lands, the paper may want to become a different paper — see
§5.

### 3.4 Data ladder — endorsed, with the ordering made explicit

The handover's central framing is correct and should be quoted at anyone who
proposes more resolution: **only new temporal samples touch the measured
constraint.** Going to 1/12° would give 1.9 billion windows from the *same
2,417 bins* at nine times the disk.

Recommended order:

1. **Move the 32 `rg_*` channels to a 1° sidecar** — frees ~27 GB, more than
   every proposed import combined, and R0 scores them null at every lead. Not a
   data import at all.
2. **The 2025–2026 continuation** (+4.5 % end-bins, nearly free) and **the
   backward extension toward 1958** (EN4.2.2 / IAPv4, ~3 GB). The only
   observationally-constrained additions that touch the wall. Run the backward
   extension **as a measurement first**: build it, train a small-tier head, ask
   whether the step-2,000 wall moves.
3. **ERA5 forcing via a CDS account** (+5 GB net). Buoyancy forcing is entirely
   absent from the tensor today and it is the mechanism the subpolar precursor
   is supposed to run through. This is *missing physics*, not more data.
4. **The statics and RAPID's unused files** (~3 MB) — bathymetry, slope, mask,
   `f`, `β`, distance-to-coast, `moc_vertical`. Highest information per byte in
   the whole ladder. An afternoon.
5. **GREP 3-member** as low-bias augmentation, again run as a measurement.
6. **CMIP6 (family 6) — codec pretraining only, never the forecaster.** Chris's
   distillation objection is right: CMIP6 AMOC strength spans ~10–30 Sv against
   RAPID's ~17, and AMOC variability is among the most model-dependent
   quantities in the archive. The one thing we are trying to predict is the
   thing these models are least trustworthy about. Learning what ocean states
   *look like* is far less bias-sensitive than learning the forward map.
   The registered control is mandatory: embed the **reanalysis** with the same
   37 channels masked, or "pretraining helped" cannot be distinguished from
   "the fine-tune adapted to its pretraining channel set".

---

## 4. Proposed next steps, in execution order

**Nothing new is trained until the nulls read out.** The programme's history is
that evaluation errors outran modelling gains by a wide margin.

**Stage A — evaluation quartet (already queued by Opus, endorsed unchanged)**

1. **Roll the step-2,000 checkpoint.** Cheapest untried thing in the programme.
   Every rolled number to date comes from 198,000 steps *past* the held-out
   minimum. This alone could change the answer to Q1.
2. **Apply the amplitude calibration** to already-computed states. Pure decoding
   change; the identity `msss = 1 − (1 + a² − 2a·ACC)` predicts −0.439 → +0.019.
   If it fails, the identity is wrong somewhere and that is worth knowing.
3. **Mirror `temporal_e060a.pt`** (30,425,836 bytes) from
   `gs://earth-tpu-staging/runs/e060a-8m/` to the `model-checkpoints-v1` release
   so the roll evaluator can read it. Recorded as the one blocker; **it is not
   blocked** — the GCS key at `/home/claude/.gcp/sa.json` reads the bucket. Use
   `scripts/publish_heads.mjs`'s verification against the head's own args; do
   not skip it.
4. **Nearest-analogue retrieval baseline.**

**Stage B — this session's additions**

5. **Damped persistence + LIM baselines** on the same protocol (§3.1).
6. **Codec contamination check and ruling** (§1.3).
7. **Blocked-CV harness + block-bootstrap CIs** (§3.2).
8. **Quarantine note in `paper.tex`** (§3.3).

**Stage C — only after A and B read out**

9. Terminal-holdout retrains at 7.6M under the frozen protocol, 5 seeds.
10. Longitude-holdout retrain (Q4 — R0 could not measure spatial generalisation
    because the pentad r2 tensor has no longitude hole).
11. Data ladder items 1–4 from §3.4.

**Human-only items** (cannot be done from a session):

- Add the **`HF_TOKEN` Actions secret** (Settings → Secrets). The repo has no
  Actions secrets at all and the PAT returns 403 trying to write one. Until
  then family 6 lives on one rented disk (`gpu-box-35586926`, stopped, disk
  intact — **do not destroy it**).
- Optionally **enable the Cloud Quotas API** so TPU waves can run in parallel
  instead of sequentially.

---

## 5. The open decision — this is the one that shapes everything

**Does AMOC transport at 26.5°N stay the headline, or does the headline pivot to
field-level subseasonal-to-seasonal skill (SST / SSH / MLD), with AMOC demoted to
a bounded secondary read-out?**

**Recommendation: pivot.**

The argument: at ~9 effective starts, "can this predict AMOC?" is not resolvable
with this data, and no architecture change or CMIP6 corpus alters that. The two
questions the data *can* answer are:

- **Where does learned skill beat classical nulls, and out to what lead?**
  R0 already localises signal (SST to 90 d, SSH at short lead) and the effective
  sample count for fields is in the thousands.
- **What bounds does 43 years of reanalysis place on North Atlantic
  predictability?** A rigorous limits-and-power analysis — which is the honest
  version of what the last three weeks discovered by accident, and is
  publishable in its own right.

Under the pivot, AMOC is still reported, with intervals, and the programme's
contribution to it is the *bound* rather than the forecast.

**This decision was put to Chris at the end of the chat session and is
unanswered.** A Cowork session should not assume either way; if it is still open,
Stage A is safe to execute regardless — the quartet is protocol work that both
paths need.

---

## 6. Infrastructure notes carried forward (from the Opus handover, condensed)

- **Push to `blauewelt/earth` fails through the sandbox git proxy (403).** Use
  `node scripts/git_api_push.mjs --token-file /home/claude/.gh_pat --repo
  blauewelt/earth --branch main --range origin/main..HEAD`. Other sessions push
  to `main`; rebase first, **never force**.
- **Credentials in files, never argv.** GitHub PAT `/home/claude/.gh_pat`;
  Vast `/home/claude/.vast_key`; GCP `/home/claude/.gcp/sa.json`; HF
  `/home/claude/.hf_token`; CMEMS env-only by policy. On a subscription without
  this project's docs they must be re-supplied.
- **`ml-train.yml` is at the 25-input ceiling** — a 26th input makes the
  workflow unparseable. New capabilities go in as *values* on an existing input.
- **`mint_token()` in `scripts/tpu_box.py` returns a TUPLE — take `[0]`.**
- **No shell reaches a running Vast box**; work on it goes through a dispatched
  Actions job. Boxes often exit after provisioning — check
  `node scripts/gpu_box.mjs list` and `start <id>` before blaming a queue.
- **Disk is the most common cause of death** (#508, #509 both died on a full
  100 GB disk, one before hygiene could run).
- **TPU node names are the resume key.**
- One 200k-step run is ~16.5 h on a TPU v5e-4; spot quota is 4 cores/zone and
  three of five reachable zones have no capacity, so waves run sequentially.
  This shapes what is answerable per week more than any scientific
  consideration.

---

## 7. Working principles worth restating

From `ml/CLAUDE.md` and from how this correction was found:

- A backup is only real if the restore works (§0.2).
- Preconditions fire before the expensive part (§0.3).
- Write the experiment entry **at dispatch**, not after (§1).
- Check `wc -c` on a plan after writing it — `docs.html` registers the path, not
  the size. (`E059_holdout_window.md` was committed empty twice.)
- **Assume the evaluation stack is still wrong somewhere.** It has been wrong
  twice — once structurally (the pool), once semantically (calling `msss_clim`
  an AUC) — and both survived months of use. Verifying a metric is ordinary
  work, not paranoia.
- Record failed predictions as failed. The 7.6M-within-0.02 prediction failed at
  0.051 and is recorded in `ml/EXPERIMENTS.md#e-060-read`; a future session
  should not find a softened version anywhere.

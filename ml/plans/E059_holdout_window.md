# E-059 · The memorization-controlled head: retrain E-051 with the held-out years actually held out

Registered 2026-08-28, before dispatch. Chris: *"Sounds good, let's fix
training."*

## The bug this experiment exists to price

Discovered 2026-08-28 when Chris asked where measured 2009 leaks into #503's
flat skill profile. The stage-2 loss is dense over the window — `win_ztgt`
supervises the next-step prediction at every one of the K frames — while the
pool excluded a window only when its FINAL target `t+1` was held out. So a
window ending in the ~K bins after a holdout year teacher-forced that year's
measured transitions into the weights (of order tens of times per pixel over
a 400k-step run) and read its bins as context besides. The held-out year was
held out as an ENDPOINT, not as an experience. Every archived stage-2 head,
monthly and pentad alike, was trained under that pool — including the
monthly 0.939 corridor champion (K=24 months is the same two-year span).

The fix is `--holdout-scope window` (commit c25f6ff): at that scope a window
is eligible only if NONE of the bins its forward pass touches — frames,
per-frame targets, scored reach — is held out, and the pool brute-force
recertifies that fact before any weight is trained. The default stays
`endpoint` so every archived number remains reproducible; the flag is
recorded in `stage2_config.holdout_scope` so an artefact self-describes.

Design note — why full-window exclusion rather than a loss mask: a loss mask
would stop the year being a TARGET but still feed its measured bins in as
INPUT (windows ending after the year contain it as context), and the eval
targets appearing in training inputs is still contamination for a claim as
scrutinised as forecast skill. `window` makes the year invisible to training
outright, at a measured cost of ~13% of the pool (see the prediction below).
If a later decomposition of the memorization term into target-leak vs
input-leak is wanted, the loss-mask variant is the intermediate arm; it is
not needed to answer the headline question.

## The experiment

**E-059 · retrain the E-051 configuration bit-for-bit except the pool ·
params 206.659M · stage stage-2 · data family4_na025_pentad_r2 (37e146384b) ·
arch 1024×16, K 144, stencil 145 ring spiral:111-4444-0.71-0.5, znoise 0.7,
grad-clip 128, seed 0, frozen run-415 codec, published Z · steps two phases
matching the rolled artefact's own two `stage2_config` records — 200k at lr
1e-3 halflife 40k warmup 2k, then 200k→400k at lr 4e-4 halflife 100k ·
resume none (fresh) · framework JAX/v5litepod-4, node `e059-window` ·
THE ONE CHANGE: `--holdout-scope window`.**

One variable against the head #503 rolled. The two-phase LR history is
replicated deliberately: the comparison artefact is the 398k head, and its
schedule is part of its identity.

**First-minutes checks (exact, computed in advance):** `resolved knobs`
must show `holdout_scope window`; the pool print must appear with its
certificate; and the pool arithmetic is predicted here so a deviation stops
the run before it spends: endpoint end-bins 2,779 (= 3,142 − 219 − 144,
measured on E-051); window excludes additionally 144 (after 2009) + 144
(after 2017) + 74 (after 2023, truncated by the axis end at bin 3,140) =
362, so **2,417 end-bins** and `train_windows` = 86,698 × 2,417 =
**209,549,066**. If the node prints anything else, the 2023-block
arithmetic or the fix is wrong — investigate, do not train.

**Registered readings, in order:**

1. **One-step ratio at 200k and 400k vs E-051's 0.0330 / 0.02981** — same
   denominator by construction (`ev_m` untouched; `val_persistence` must
   read 21.44621 exactly). PRE-REGISTERED INTERPRETATION: the val targets
   are holdout bins, and under the OLD pool those exact transitions were
   trained on as within-window targets — so E-051's one-step number is
   itself suspect. **The gap between E-059 and E-051 at matched steps IS
   the memorization term at h=1, measured directly.** If E-059 ≈ 0.0298,
   memorization contributed ~nothing at one step; a materially worse ratio
   is not a regression, it is the honest number.
2. **The roll** (same protocol as #503: horizon 73, 3 starts/holdout year,
   day-matched): the headline is the SHAPE — genuine forward skill decays
   with lead; #503's head read flat 0.946 at 365 d. Corridor
   `horizon_auc_daymatched` vs #503's 0.944, with both of #503's caveats
   inherited (no pentad gate reference; battery required).
3. **The battery** on the E-059 head, and beside it the shortened battery
   re-roll of the OLD head (dispatched with this plan): the old head's
   future roll — beyond the record's end, where nothing existed to
   memorize — is the direct recall test that needs no retrain.

**Falsifiers.** If E-059's roll is ALSO flat at high skill, memorization was
not the mechanism and the flat profile needs another explanation (the
anomaly field's own persistence structure would be the next suspect — check
`msss_pers`, which no innocent story puts at 0.966 at 365 d). If E-059's
skill decays to climatology within a few months, the pentad programme's true
state is that its rolls were never certified — and the monthly 0.939
champion inherits the question in full.

**Cost.** Phase 1 ≈ 12.6 h on a v5litepod-4 (measured: E-054a did 200k in
45,336 s), phase 2 the same; spot ≈ $60 total, on-demand ≈ $121. Roll ≈ $4.
Spot-first per the standing rule; us-west1-c's spot quota is held by
E-054b's node, so the ladder starts at us-west4-a.

## Follow-ups this plan does NOT cover

- **The monthly champion retrain** (`xl144-nolonhold` at scope=window on the
  monthly tensor) — after E-059 reads out; if the pentad memorization term
  is large, the monthly one is priced the same way.
- **The codec's residual exposure**: `train.py`'s neighbour term (dt ±1)
  predicts the measured value one bin across a holdout boundary — 6 boundary
  bins of 3,142, one step deep, in a per-pixel autoencoder that is never
  rolled. Recorded as negligible for the roll question; a
  `--holdout-scope`-style clamp on the neighbour draw is a cheap later
  change. E-059 deliberately reuses the FROZEN run-415 codec so the retrain
  is one-variable against E-051.
- **Flipping the default** to `window` for all future arms — Chris's call,
  after E-059 lands, in a coordinated change with the recipes.

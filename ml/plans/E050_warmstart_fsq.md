# E-050 · Warm-start quantization: the trained continuous codec, lattice switched on

**E-050 · resume the finished E-049a continuous d_z-6 pentad codec (#480 →
`run-480__pixelmae.pt`) with the FSQ lattice AND the ln bound switched ON at
resume, via `--fsq-warmstart` · params 37.956M (measured identical on both
arms — the lattice and the bound are parameter-free) · stage encoder
(resume + 60k warm steps) · data `family4_na025_pentad_r2` · arch 512×12
d_z 6 patch 1, FSQ [8,8,8,5,5,5] (2¹⁶ = one 16-bit token per pixel-bin),
ladder auto (fits REBASED to +50/+200/+2000/+20000/+60000), bound ln ·
steps×batch 260,000×512 (= 200,000 resumed + 60,000 warm) · resume
`run-480` (REQUIRED, `!`-prefixed).**

TL;DR — every cold-start codec-side FSQ run has collapsed or degenerated
(e048a, #481/#482 to a constant encoder; run-455 to a sign code), while
quantization ON AN EXISTING z works (A9: 0.4916 < 0.5056) and lattice z
out-forecasts continuous z at the registered E-046 instrument (#477:
0.4394). E-050 asks: does giving the lattice a TRAINED encoder — directions
already spread, distribution already settled — produce the road-B token
without the collapse? Chris approved the line 2026-08-25 (~15:30Z, b3ee36a).

## 1 · Hypothesis and falsifier (registered before dispatch)

**Hypothesis.** Warm-started onto #480's settled z distribution, the
[8,8,8,5,5,5] lattice with the ln bound trains WITHOUT direction collapse,
and the resulting one-token codec passes E-049b's registered audit: the
decoder ceiling (E-019b1 decoder-only retrain against frozen tokens)
reconstructs the fast channels on Argo-free bins at FVU inside the
month-block codec's accepted 9–19% band.

**Falsifier A (the collapse repro).** Fit `prequant_std_med` falling toward
~0 (the #482 signature: 0.638 → 0.005 by 7.5k) or persistence z-MSE
trending to exactly 0 in the warm phase ⇒ warm-start does NOT repair the
cold-start disease; the next lever is the bound (RMS-only, no
mean-centering — gated on the #482 corpse diagnosis), not more steps.

**Falsifier B (the capacity verdict, inherited verbatim from E-049b).** Any
fast channel at ~100% FVU at the decoder ceiling on Argo-free bins ⇒ 16
bits cannot carry a bin even from a trained encoder; the pre-registered
next rung is TWO tokens per bin (d_z 12 = the same levels twice), never a
larger vocabulary.

**Declared confound.** The warm arm totals 260k steps vs #480's 200k, so
(E-050 − #480) is quantization tax PLUS 60k extra continuous-equivalent
steps. The falsifier is absolute (the FVU band); a PASS near the band edge
buys a 260k continuous control before any tax number is quoted.

## 2 · Health signals, in first-minutes order

1. The trainer's `FSQ WARM START` banner naming
   `fsq_warmstart_from=run-480.pt@200000` — the guard hole opened for
   exactly this transition and nothing else.
2. First rebased fits (+50, +200): `prequant_rms` 1.0 AND `std_med`
   input-dependent (O(0.1–1)) — NOT the #482 slide toward 0.005.
3. `ml/fsq_usage.py` effective bits on the first milestone: a healthy
   16-bit token reads well above the ~6-bit sign-code floor run-455 wore.
4. `collapse_r` stays 0 (uncalibrated on lattice z; #481's false-fire
   analysis stands); the fit records + effective bits are the monitor, with
   #480's own fit trace as the healthy reference.

## 3 · Protocol

Dispatch = #480's INPUTS_JSON with `window: recipe:f4r2-40M-dz6-fsq65k-warm`,
`resume: "!run-480"`, `steps: 260000`, pinned to `gpu-box-32966687` — the
box that trained the parent (#480 finished there 05:37Z, tensor and caches
warm, box running idle at dispatch time; the resume seed pulls
`run-480__pixelmae.pt` from the release, promoted 2026-08-26 from Actions
artifact `pixelmae-480`). Train ~4.5 h (0.252 s/step measured on #482's
sibling 4090) + the probe ladder; then the E-049 audit pair (`recon_eval`
full-visibility FVU + `recon_decoder` ceiling) on the per-(t,pixel)
Argo-bin split, exactly as adapted at 5f40551. §3b applies in its harder
form: first result at a new configuration — it buys its replication before
any number is a level.

**Corpse forensics available for a Falsifier-A follow-up:** #481's ~10k
guard-killed checkpoint is ALREADY published (`rescued-orphan-latest-482.pt`
on model-checkpoints-v1 — the rescue step at #482's start, labelled by the
rescuing run's number); #482's ~15k fully-collapsed corpse sits on
`gpu-box-42005419`'s stopped disk (Vast 48478309) and publishes on that
box's next job. Pre-LN vs post-LN constancy on either decides ln vs
RMS-only.

## 4 · Risks

The rebased +50 fit measures a distribution the ln bound has only just been
applied to (run-480 trained UNBOUNDED — the bound normalizes its z at
resume), so the earliest lattice is fitted to a freshly renormalized
distribution; the +2000 and +20000 refits are the correction path, and
e048a2's lesson (each fitted lattice outgrown in ~10k when the scale
drifts) is why the fit schedule extends into the warm phase. If the bound
itself proves the poison (Falsifier A with the corpse diagnosis blaming
mean-centering), the RMS-only bound is the pre-named alternative.

# E-064 · A quantized token as the forecasting substrate, under the clean pool

**Written 2026-09-02 ~10:2xZ, before dispatch.** Chris: *"Another agent has
had good results with FSQ (30 bits). Are those in the paper? If not, can you
run FSQ and add those results?"*

## 1. What is already known, and what is not

Codec-side quantization is measured and clean (the codec's held-out years
are held out outright): the warm-started 16-bit lattice (E-050, run-485:
d_z 6 through FSQ [8,8,8,5,5,5], LayerNorm bound, 60k warm steps on the
finished continuous d_z-6 codec run-480) holds input-dependent activations
throughout, pays an 18 % reconstruction tax (`loss_rec` 0.270 vs 0.229) and
reads RAPID at 0.588 [0.534, 0.641] against the continuous 0.579 [0.525,
0.627] through the unpooled head — the same at n = 1. Every cold-start
lattice collapsed (E-049b, twice) or degenerated to a sign code (E-046,
run-455). None of this is in the current report yet; §2 of the report gets
one paragraph from it whatever E-064 returns.

What is NOT known is whether the token forecasts. The only stage-2 numbers
on the token (E-056a, #504/#507: one-step 0.539 at z-noise 0.7 and 0.510
dose-matched at 0.12, against the continuous 0.506, all at K = 24, 1024×16)
were trained under the endpoint pool, i.e. with the held-out years
teacher-forced, and are retired with every other pre-c25f6ff stage-2 number.

The "30 bits" the other agent used is not described anywhere in this
repository or the project; this plan runs the lattice that exists (16 bits)
and names the 30-bit codec as a follow-up once its levels are known.

## 2. The arms

Both arms are the 7.6M small-tier head (256×8, K 144, stencil 145 ring
`spiral:111-4444-0.71-0.5`, batch 256, lr 1e-3 expdecay halflife 40k warmup
2k, grad-clip 128, seed 0, `--holdout-scope window`, 20,000 steps, val every
200), trained with the torch trainer on one Vast 4090, sequentially, so the
pair shares box, code and pool certificate.

| arm | codec (frozen) | z | z-noise | control for |
|---|---|---|---|---|
| **E-064a** | run-485 (E-050 warm FSQ, d_z 6, 2¹⁶ codes) | published token Z `Z_867532fe7b_37e146384b` | **0.12** (dose-matched: 0.7 × 0.15116/0.87878, the #507 arithmetic) | the question |
| **E-064b** | run-415 (continuous d_z 32) | published `Z_8b639abe36_37e146384b` | 0.7 | E-060a's torch twin — the same configuration the JAX trainer ran on TPU |

E-064b exists because E-060a was trained by the JAX port; a torch/JAX pair
at this tier has never been measured, and a comparison across trainers would
otherwise carry an unmeasured backend term. Its number also gives the
programme a torch-format early-stop checkpoint of the continuous 7.6M head
(milestones at 1,000 / 1,200 / 1,400 / 1,600 / 2,000 / 3,000), which the
reboot plan's step 1 ("roll the step-2,000 checkpoint") needs and which the
TPU run did not keep.

Header lines (§0d):

- **E-064a · stage-2 head on the 16-bit token, clean pool · params 7.6M
  head over frozen run-485 (37.956M, nothing trains) · stage stage-2 · data
  family4_na025_pentad_r2 · arch 256×8 K 144 stencil 145 ring spiral, d_z 6
  · steps×batch 20k×256 · resume run-485@260k (release).**
- **E-064b · the same head on the continuous d_z-32 z, torch twin of E-060a
  · params 7.6M head over frozen run-415 (37.976M, nothing trains) · stage
  stage-2 · data family4_na025_pentad_r2 · arch 256×8 K 144 stencil 145 ring
  spiral, d_z 32 · steps×batch 20k×256 · resume run-415@197428 (release).**

## 3. Pre-registered readings

The instrument is the held-out one-step z-MSE / persistence (scale-free, so
the token scale and the continuous scale compare directly), read as (i) the
**minimum over the 20k curve** and the step it occurs at, and (ii) the value
at 20,000. E-060a (JAX) reads 0.6095 at step 1,200 and 0.692 at 20k.

- E-064b within **0.02** of E-060a on both readings → the torch and JAX
  trainers agree at this tier and E-060a's numbers may be compared with
  torch arms directly. Outside 0.02 → a backend term exists and every
  cross-trainer comparison must carry it.
- E-064a's minimum within 0.02 of E-064b's → **the 16-bit token is a
  competitive substrate** at 5 % of the state size; the token road opens for
  the clean re-ranking. Worse by more than 0.02 → quantization removed
  forecastable information, stated as a level only after a second seed.
  Better by more than 0.02 → a genuine regularisation effect; the same rule.
- Both arms must reproduce the pool certificate (2,417 end-bins, 0 touching
  a held-out bin, 209,549,066 windows) and E-064a's monitor must read
  `input_znoise_rel_pers` ≈ 0.15 — if it does not, the dose is wrong and the
  arm is re-run before anything is read.
- Lead-decay is not read here: this is a one-step gate. A roll of the
  early-stop checkpoint follows only if the token passes it.

Single seed each. The only pentad one-step pair on record (E-044b, 206M at
200k) reads |Δ| 0.001; the 7.6M tier has no pair, so the 0.02 bar is a
working threshold and any sub-0.02 difference is written as a consistency.

## 4. What it costs

E-050's box measured ~0.2 s/step for a 40M codec; a 7.6M head at K 144 on a
4090 should run under 0.3 s/step → ~1.7 h per arm plus the probe ladder each
run carries (~0.5 h) and the Z pull for E-064b (16 GB). ≈ 5 h of a $0.32/h
box, under $2.

## 5. Follow-ups this plan names but does not run

- **The 30-bit lattice.** With the levels known (six dims at 32 levels is
  the obvious 2³⁰ on d_z 6; ten dims at 8 levels is the other), it is one
  E-050-shaped warm start — `fsq_warmstart` on run-480 with the new
  `fsq_levels` — followed by the same two-arm gate.
- **The roll.** If E-064a passes, its early-stop milestone goes through
  #516's battery (`sroll:`), decoded through the run-485 codec, and the
  report's Table 2 gains a column.

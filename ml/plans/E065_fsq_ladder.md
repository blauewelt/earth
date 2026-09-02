# E-065 · The lattice ladder: 16 → 24 → 30 bits per pixel-bin

**Written 2026-09-02 ~10:2xZ, before dispatch.** Chris: *"Try the following
FSQ: 24-Bit FSQ (Levels [8 x 8] — 8 8 8 8 8 8 8 8 = 16.8M states, dz = 8).
And then 30bits (Levels [8 x 10])."*

## 1. What is being built

Two more quantized codecs beside E-050's 16-bit one, so the token's
forecasting value (E-064's question) can be read as a function of the
bits it carries:

| lattice | d_z | levels | codes per pixel-bin | parent (continuous) | warm run |
|---|---|---|---|---|---|
| 16-bit (E-050, exists) | 6 | 8,8,8,5,5,5 | 2¹⁶ = 65,536 | run-480 | run-485 |
| **24-bit** | 8 | 8×8 | 2²⁴ = 16.8M | E-065a (new) | E-065b |
| **30-bit** | 10 | 8×10 | 2³⁰ = 1.07G | E-065a2 (new) | E-065b2 |

Every lattice is built the way E-050 was, because that is the only way a
lattice has survived on this tensor: **a continuous codec at the target
d_z is trained to 200k steps, then resumed with the lattice and the
LayerNorm bound switched on for 60k warm steps** (`--fsq-warmstart`; the
lattice and bound add no parameters, so every trained weight loads and only
the bottleneck's function changes). Cold starts collapsed twice (E-049b) and
the unbounded d_z-32 lattice degenerated to a sign code (E-046). There is no
continuous parent at d_z 8 or 10, so the parents are the first two runs.

Recipes: `f4r2-40M-dz8` / `f4r2-40M-dz10` (parents: `f4r2-40M-dz6` with
d_z changed, nothing else) and `f4r2-40M-dz8-fsq24-warm` /
`f4r2-40M-dz10-fsq30-warm` (E-050's warm recipe with d_z and `fsq_levels`
changed, nothing else — same auto ladder, same fit schedule, same `ln`
bound, `collapse_r` 0).

## 2. Readings, in order

1. **Parents** (E-065a, E-065a2): `loss_rec` and the unpooled RAPID head
   against run-480 (d_z 6: 0.229 / 0.579) and run-415 (d_z 32). These are
   the width-tax controls; a parent that reads worse than run-480 at a
   wider bottleneck has a training problem, not a width one, and its
   lattice is not built.
2. **Warm lattices** (E-065b, E-065b2): E-050's Falsifier A — the fits at
   +50/+200/+2000 warm steps must stay input-dependent (`prequant_std_med`
   O(0.1–1), never → 0.005) with `prequant_rms` 1.0; effective bits from
   `ml/fsq_usage.py` must sit well above a sign code (8 and 10 bits
   respectively). Reconstruction tax against the parent, and the head
   probe, reported beside E-050's 18 % / 0.588.
3. **The gate** (the E-064 protocol, one arm per lattice): the 7.6M head,
   K 144, `--holdout-scope window`, z-noise dose-matched to
   `input_znoise_rel_pers` ≈ 0.15 on each lattice's own scale (the #507
   arithmetic, from a short calibration read of the monitor), 20k steps,
   read at the curve minimum and at 20k against E-064b. The result is a
   three-point curve — 16 / 24 / 30 bits — against the continuous 32-d
   embedding.

Pre-registered: if the token's one-step ratio improves monotonically with
bits and the 30-bit lattice is within 0.02 of the continuous twin, the
token road is open and the bit count is the knob; if all three lattices sit
at the same distance below the twin, quantization costs a fixed amount the
bits do not buy back and the loss is in the bound or the lattice geometry,
not the vocabulary. One seed per arm; the 7.6M pentad tier has no measured
pair, so every sub-0.02 difference is a consistency.

## 3. What it costs

Each parent is a run-480-shaped job: ~14 h of a 4090 for 200k steps plus
the probe ladder (~$5). Each warm run is ~4.5 h (~$1.5). Each gate is the
E-064 price (~$1). The two parents run in parallel on two fresh boxes
(`gpu-box-39184686`, `gpu-box-48397639`); the warm runs and gates follow on
the same boxes, which then hold the parent checkpoint locally. ≈ $15 and
~2 days end to end.

## 4. What must be verified, not assumed

- The parent's checkpoint must be **promoted to `model-checkpoints-v1`** as
  `run-<n>__pixelmae.pt` before the warm dispatch names it in `resume` —
  E-050's parent needed a hand promotion (the release publish did not run).
- The warm dispatch's first minutes: the `FSQ WARM START` banner naming
  `run-<parent>.pt@200000`, `fsq_levels` with the right number of dims,
  `fsq_bound ln`, and the +50/+200 fits.
- The gate's z-noise: read `input_znoise_rel_pers` from the first monitor
  record and re-dispatch with the corrected dose if it is not ≈ 0.15 — the
  dose does not transfer across lattices as a constant.

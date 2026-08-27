# FGN on TPU — porting the ε-conditioned head + fair CRPS to ml/jaxport

*Scoped 2026-08-27 ~21:25Z at Chris's direction: "Let's also get the FGN
TPU code ready for the future (there should be open source Jax code, if
not, let's just implement it). Then consider moving the 30h job and its
roll to tpu."*

Status: **PHASE 1 IN BUILD** (Opus subagent, spec below; main session
verifies). Phase 2 (ensemble roll) gated on phase 1's certificate.

## 0 · The open-source JAX reference exists, and we already audited it

DeepMind's FGN is open-sourced as WeatherNext 2
([google-deepmind/weathernext](https://github.com/google-deepmind/weathernext),
Apache-2.0, JAX) — `weathernext2/fgn.py` + `utils/dense.py`. E-057(g)
matched our torch implementation against it line by line: `crps_loss`
(unbiased, M=2) ≡ our `fair_crps2`; `LinearNormConditioning` ≡ our FiLM;
noise as one global vector into every block's norms; ε resampled per
predictor call in rollout. **The port target is therefore OUR
`ml/temporal.py` semantics, with weathernext as the arbiter for any
arithmetic question — not a code import.** The three registered torch
deviations (exact-zero film init; stock LayerNorm affine kept, modulation
on top; separate (scale, shift) per sub-layer norm → film maps to 4·d per
block) are KEPT IDENTICAL in JAX, so torch and JAX stay twins of each
other, not two children of weathernext.

## 1 · Phase 1 — training (in build)

Surface, mirroring torch exactly:

- `ml/jaxport/models.py`: FiLM conditioning on the stage-2 transformer's
  norms at the SAME sites as temporal.py — per EncoderLayer, `norm1` and
  `norm2` each get `x_norm · (1 + scale) + shift` with (scale, shift)
  from a zero-init linear of the ε embedding (film → 4·d per block), plus
  the out_norm site. ε ~ N(0,1)^k, one global vector per sample.
  Flag-off (`fgn_eps = 0`) builds no film params and is BITWISE the
  existing model — the existing equivalence certificates must stay green
  untouched.
- `ml/jaxport/train_stage2.py`: `--fgn-eps k` + `--fgn-val-members M`;
  fair-CRPS N=2 objective (two forwards per step, ε₁ ≠ ε₂, mean|xᵢ−y| −
  |x₁−x₂|/2); under MSE the flags REFUSE (same guard as torch — an
  ε-conditioned head under MSE is a meaningless arm). Telemetry writes
  the SAME record keys as torch (`stage2_val_crps`,
  `stage2_val_member_var`, `stage2_val_spread_ratio`, config keys
  `fgn_eps`/`fgn_val_members`/`stage2_loss_kind: crps2`) so status.html
  needs no new record family (§0d's status-page rule satisfied by reuse).
  **The M-member monitor eval is CHUNKED from birth** — #496 died OOM
  because torch's monitor pushed the full 4096-window batch per member;
  the fix (512-window slices) is part of the port's spec, not a later
  patch.
- ε stream: per-step ε from JAX PRNG folded from (seed, step, member),
  checkpointed via the step counter (JAX PRNG is counter-based, so exact
  resume is the fold arithmetic — simpler than torch's generator-state
  save). Cross-backend ε-stream equality is NOT required and NOT claimed
  (different RNG families); within-backend resume-exactness IS.
- `ml/jaxport/convert.py`: film params round-trip pt ↔ npz, so a torch
  FGN head can be continued/rolled on TPU and vice versa.

**The certificate (the port is unusable until this passes, CPU, $0):**
(a) forward equivalence — a torch FGN toy head converted to JAX, the SAME
ε INJECTED on both sides (bypassing both RNGs), outputs match to the
float-dispatch floor (score_section_probe's class: ≤ ~1e-4 relative;
record the measured number); (b) `fair_crps2` loss value on fixed inputs
matches torch to 1e-6; (c) flag-off purity — existing jaxport tests and
the G3 twin certificate untouched and green; (d) OOM-lesson test — the
monitor eval at a toy scale produces identical results chunked vs
unchunked.

## 2 · Phase 2 — the ensemble roll on TPU (gated on phase 1)

`ml/jaxport/roll.py` already rolls deterministic heads. Extend: M member
trajectories per start, ONE ε per (member, step) shared across all pixels
(FGN's convention, as the torch evaluator does), ensemble-mean field
through the unchanged scoring path, and the SAME new-key names as the
torch ensemble roll (`ens_prob`, `amoc_bands_ens(+_unpooled)`,
`long_dispersion`/`future_dispersion` with the variance floor) so results
are comparable by key. Acceptance: key-name parity with the torch
evaluator on a toy, deterministic-path purity, and a cross-backend
ensemble-mean corridor check on a converted toy head.

## 3 · What moves to TPU, and what deliberately does not

- **The in-flight pair #500/#502 does NOT migrate.** They land ~24Z
  08-28; no certificate can be trusted before then, and mid-comparison
  backend switches put the pair across a boundary its controls (torch:
  clean 0.6781 / znoise 0.7235) never crossed.
- **What the port is FOR:** (i) future FGN arms — winner replicates,
  bigger FGN heads — on v5e where wall-clock roughly halves (measured
  classes: v5e-4 ~4.6 steps/s at K=144/206M vs the 4090-class ~2.05
  steps/s at K=24 with two forwards); (ii) E-052.2's FGN-mode FieldDiT
  arm (already JAX — the addendum's proposal lands nearly free once the
  loss + conditioning exist in jaxport); (iii) the ensemble roll at M=8+,
  which is 8× roll compute and the first place TPU throughput really
  pays.
- **Economics, stated so the choice stays honest:** TPU spot ~$1.7/h vs
  a 4090-class box ~$0.30/h — TPU is ~2× the $ per result at ~2× the
  speed, AND the v5e spot grant is one node per zone (SPOT_LEDGER
  08-27), so GPU boxes remain the cheap wide-parallel pool. TPU FGN is
  for wall-clock-critical arms and for the roll, not a blanket
  migration.

## 4 · Cross-backend comparability rule (pre-registered)

A JAX-trained FGN number is comparable to torch-trained controls only as
a DIRECTION until a backend pair is measured: the first JAX FGN arm at a
torch-measured configuration buys a same-seed torch twin (one run), and
that pair's delta extends §3b's table as its own row. The box-effect
history (0.041 on the head k-fold from environment alone) is the reason
this is not optional.

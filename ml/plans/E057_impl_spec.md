# E-057.0 implementation spec (main-session spec; subagent implements — ml/CLAUDE.md §0b)

Repo checkout: `/home/claude/work/earth`. Registered plan: `ml/plans/E057_fgn_head.md`
(read it first). Everything here is ADDITIVE and flag-gated: with the new flag at its
default, `ml/temporal.py` must be BIT-IDENTICAL in behaviour to today — no new module
constructed, no extra RNG draw, no changed record. Torch in this sandbox is 2.13.0+cpu.

## 1 · `ml/temporal.py` changes

### 1a. New CLI flags (argparse, near --input-znoise)

- `--fgn-eps` int, default 0. k = dimension of the global noise vector
  ε ~ N(0,1)^k (FGN, arXiv:2506.10772). 0 = OFF = the exact legacy code path.
  When > 0 the training objective SWITCHES to the fair CRPS at N=2 (below) —
  one flag, because under plain MSE a noise-conditioned head learns to ignore
  ε (the conditional mean is optimal); the flag's help text must say this.
- `--fgn-val-members` int, default 8: ensemble size for the in-training
  monitoring reads.
Both must ride through the `window` `sched:` tail like `--input-znoise` does
(no workflow edit — that mechanism passes unknown flags verbatim; verify by
reading how input-znoise reaches the parser, and do the same).

### 1b. Model: ε-conditioning via per-layer FiLM on the pre-LN stream

`TemporalTransformer.__init__` gains `eps_dim=0`.

- `eps_dim == 0`: EXACTLY today's constructor — stock
  `nn.TransformerEncoder(layer, n_layers)` — so every published checkpoint
  still loads `strict=True` and the no-flag path constructs the identical
  module tree.
- `eps_dim > 0`:
  - `self.eps_embed = nn.Sequential(nn.Linear(eps_dim, d_model), nn.SiLU(),
    nn.Linear(d_model, d_model))` → conditioning vector c [B, d_model].
  - The encoder is a thin container whose state-dict keys MATCH the stock
    encoder's (`encoder.layers.N.self_attn.*`, `norm1`, `linear1`, ... ) so a
    legacy deterministic checkpoint warm-starts the trunk with
    `load_state_dict(..., strict=False)`: implement
    `class _CondLayer(nn.TransformerEncoderLayer)` overriding `forward(self,
    src, c, src_mask, is_causal)` with the explicit norm_first math:

        s1, b1, s2, b2 = self.film(c).chunk(4, -1)        # each [B, d_model]
        x = src
        x = x + self._sa_block(self.norm1(x) * (1 + s1[:, None]) + b1[:, None],
                               src_mask, None, is_causal=is_causal)
        x = x + self._ff_block(self.norm2(x) * (1 + s2[:, None]) + b2[:, None])
        return x

    with `self.film = nn.Linear(d_model, 4 * d_model)` whose weight AND bias
    are zero-initialised — so at init (and whenever film is zero) the layer
    computes the stock norm_first forward EXACTLY, and the whole model equals
    the legacy incumbent with the same trunk weights. Container class
    `_CondEncoder(nn.Module)` with `self.layers = nn.ModuleList([...])`,
    assigned as `self.encoder`, `forward(x, c, mask, is_causal)` loops layers.
  - `forward(self, z_seq, month_seq, static_ctx, eps=None)`:
    - `eps_dim == 0` and `eps is not None` → `ValueError`.
    - `eps_dim > 0` and `eps is None` → `ValueError` with a message saying an
      FGN head cannot be run without its noise vector — THIS is the guard that
      makes `rollout_spatial.py` refuse an E-057 head loudly instead of rolling
      it clean (rollout builds from checkpoint args and calls with 3 args).
    - otherwise c = self.eps_embed(eps); pass c through the cond encoder.
    The 3-positional-arg call signature of every existing caller must keep
    working unchanged.

### 1c. Model construction and refusals in `main()`

- Construct with `eps_dim=a.fgn_eps` (constructor arg only when > 0 is fine).
- Refuse (SystemExit, before anything expensive — §0.3) when `a.fgn_eps > 0`
  and any of: `a.direct` non-empty, `a.unroll != 1`, `a.unroll_wide > 0`,
  `a.milestone_steps`... (milestones are fine, do NOT refuse those). Also
  refuse `a.fgn_eps < 0` and `a.fgn_val_members < 2`.
- ε generator: `eps_gen = torch.Generator()` (CPU), seeded
  `a.seed * 1000003 + 57` at start; ALL training ε are drawn from it on CPU
  (`torch.randn(B, k, generator=eps_gen)`) then moved to TDEV — device-
  independent stream. Its state is saved in every checkpoint dict as
  `"eps_gen"` (bytes of `.get_state()`, e.g. `.numpy().tolist()` like
  torch_rng) and restored on `--resume-temporal`. When `fgn_eps == 0` nothing
  is created and nothing extra is saved (bit-identical legacy artefacts).

### 1d. Training loss (fgn mode)

In the training loop, where today `pred, hid1 = model(zseq, mseq, sctx)` and
`l_base = (pred - ztgt).pow(2).mean()`:

- fgn mode: draw eps1, eps2 [B, k] (two draws from eps_gen); two forwards on
  the IDENTICAL context (same zseq — including the same --input-znoise
  corruption if any, applied ONCE before both forwards; same mseq/sctx):
  `p1, hid1 = model(zseq, mseq, sctx, eps=eps1)`,
  `p2, _    = model(zseq, mseq, sctx, eps=eps2)`.
- `l_base = fair_crps2(p1, p2, ztgt)` where, elementwise over the same
  [B, K, d_z] tensor the MSE averaged over:

      fair_crps2(x1, x2, y) = (0.5 * ((x1 - y).abs() + (x2 - y).abs())
                               - 0.5 * (x1 - x2).abs()).mean()

  (this IS the fair estimator at M=2: term2 divisor 2·M·(M−1)=4 over the
  2 ordered pairs → |x1−x2|/2). `hid1` (member-1 hidden) feeds nothing in fgn
  mode except shapes — the direct/unroll paths are refused.
- Keep the record key `stage2_loss_base` = this CRPS value; add
  `"stage2_loss_kind": "crps2"` once to the `stage2_config` record (new keys
  only, never changed ones).

### 1e. Monitoring / validation (fgn mode)

At setup, draw a FIXED eval ε bank `eps_val [M, k]` (M = a.fgn_val_members)
from a fresh generator seeded `a.seed * 1000003 + 58` (NOT eps_gen — the
training stream must not depend on how often eval runs). At each log point,
under `torch.no_grad()`, forward the monitoring batch M times (loop, chunk if
needed) → ens [M, n, d_z] of last-position predictions; log NEW keys beside
the existing ones:

- `stage2_val_zmse` (existing key) = MSE of the ENSEMBLE MEAN vs mon_ztrue —
  documented in a comment as the fgn-mode meaning (best point estimate; the
  legacy meaning is untouched when fgn is off).
- `stage2_val_crps` = torch fair CRPS of the M members (use the estimator
  from 1d generalised to M members, or compute via probscore on .cpu()
  numpy — numpy is fine here, it is 100 evals per run).
- `stage2_val_member_var` = mean over elements of the per-element member
  variance (ddof 0) — the ε-collapse telemetry; a slide toward 0 is the
  collapse signature and MUST be visible on the live branch.
- `stage2_val_spread_ratio` = spread/error ratio with the (M+1)/M correction
  (mirror ml/probscore.spread_error's formula).
- `stage2_amp` = ensemble-mean amp (same formula, ensemble mean in place of
  the single forward).
None of these keys appear when fgn is off. NaN is never written (§5.22): if a
value is non-finite, omit the key for that record and print a warning.

### 1f. Everything after the training loop (evals, probes, final save)

Every remaining `model(...)` call site (eval 1 z_t+1 / chan_t+1, the light
probe, rapid_probe paths, `_chunked_forward` callers) must work in fgn mode:
give them the REPRESENTATIVE member ε = zeros(1|B, k) (fgn's distribution
centre), via a small helper so the choice lives in ONE place, and record
`"fgn_eval_eps": "zeros"` in the `stage2_config` record. (A zeros ε through
zero-init film is exactly the legacy computation at init; after training it
is the centre member — fine for the POOLED legacy trend instruments, and the
honest ensemble read-outs are 1e's new keys.) `stage2_result` gains
`"fgn_eps": a.fgn_eps` (and nothing else changes in it).

The three `torch.save` sites add `"eps_gen"` state when fgn is on (see 1c).

## 2 · `tests/test_fgn_head.py` (new; CPU; `python3 tests/test_fgn_head.py`)

Style: follow tests/test_resume_temporal.py / test_direct_heads.py (plain
python, asserts, exact identities per ml/CLAUDE.md §4.9). Tests:

1. **Init identity, bitwise.** Build legacy `TemporalTransformer` (seed s) and
   an `eps_dim=8` twin; copy the legacy state_dict in (strict=False; assert
   the ONLY missing keys are `eps_embed.*` and `*.film.*`). In train() mode,
   same inputs, any ε: outputs `torch.equal` legacy's. Also run both in
   eval() mode: assert torch.equal; if the stock fast-path makes eval differ,
   assert max|Δ| == 0 first, and only if genuinely impossible pin ≤ 1e-6 WITH
   a comment naming the fused kernel as the cause (measure, don't assume).
2. **fair_crps2 identities.** Torch loss == ml/probscore.crps_ensemble
   (fair, M=2) on shared random arrays to 1e-12; M=1 fallback == MAE (test
   the M-member val estimator at M=1 if implemented, else skip); CRPS of
   identical members == MAE exactly.
3. **ε stream.** Same seed → identical eps draws and identical 3-step training
   losses across two fresh runs of a tiny toy; different seed → different.
4. **Resume bitwise.** Toy end-to-end through the REAL trainer path if
   affordable (CKPT_DIR_OVERRIDE + a tiny synthetic npz + tiny codec ckpt the
   way existing toy tests do it), else a faithful harness like
   test_resume_temporal.py extended with eps_gen: train 5 + save + resume 3
   == 8 straight, all parameters bit-identical AND the eps draws after resume
   bit-identical.
5. **The shared-coin toy (load-bearing).** P=32 "pixels", d_z=4: latent
   pattern g_p ∈ {±1}^{d_z-ish}; law z_{t+1,p} = z_{t,p} + s_t · pattern_p
   with ONE fair coin s_t per time shared by all pixels. Train a tiny fgn
   head (d_model 16, 1 layer, K=4, eps_dim 4, a few hundred steps, Adam) on
   per-pixel windows (conditional mean = persistence; the two futures are
   field-coherent). Then sample M=64 members at eval: for each member draw
   ONE ε shared across all P pixels, predict each pixel, compute the
   field-mean residual sign coherence |mean_p sign(residual_p · pattern_p)|.
   Assert: (a) mean coherence over members ≥ 0.8 with SHARED ε; (b) with
   INDEPENDENT per-pixel ε it is ≤ 0.35 (the factorized floor at P=32 is
   ~1/√32 ≈ 0.18); (c) the fgn head's fair CRPS (per-member ensemble,
   probscore) beats the SAME architecture trained with plain MSE (legacy
   path) scored as a degenerate M=1 ensemble, by ≥ 20%. Fix every seed; run
   it at 2–3 seeds while developing and pick thresholds that pass all with
   margin, then pin ONE seed in the committed test.
6. **Refusals.** fgn+direct / fgn+unroll>1 / fgn+unroll-wide refuse (drive
   main() argv-level if the toy harness allows, else test the guard function);
   `forward(eps=None)` on an eps model raises; `forward(eps=...)` on a legacy
   model raises.
7. **No-flag purity.** With `--fgn-eps 0` (default), the constructed module
   tree has no eps modules, `vars(a)` still round-trips into the legacy
   constructor, and a checkpoint saved by the toy run carries NO `eps_gen`
   key.

## 3 · What NOT to touch

`read_sv`, the gate, `rollout_spatial.py`, any pooled read-out, the collapse
guard, `ml-train.yml` (25-input ceiling — the flags ride the sched: tail),
`probscore.py`, any existing test. No behaviour change of any kind with the
flag off is the acceptance bar for the whole diff.

## 4 · Definition of done

`python3 tests/test_fgn_head.py` green; `python3 tests/test_resume_temporal.py`,
`python3 tests/test_direct_heads.py`, `python3 tests/test_e022_stencil.py`
still green (they exercise the legacy path); a unified diff of ml/temporal.py
that a reviewer can read top to bottom; NO commit (the main session commits).
Report: files changed, test output verbatim, any deviation from this spec with
its reason, and the measured answer to test 1's eval-mode question.

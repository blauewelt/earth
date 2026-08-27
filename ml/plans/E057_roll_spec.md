# E-057 ensemble-roll implementation spec (main-session spec; subagent implements — ml/CLAUDE.md §0b)

Repo checkout: `/home/claude/work/earth` (torch 2.13.0+cpu). Read first:
`ml/plans/E057_fgn_head.md` (the plan; E-057.1's roll read and E-057.3's
dispersion read are what this diff enables), EXPERIMENTS.md#e-057 (what
E-057.0 built), `ml/temporal.py` (the fgn head: `--fgn-eps`, `_CondLayer`,
`fair_crps2`/`fair_crps_ens`, `fgn_eval_eps`, forward(eps=...) contract),
`ml/probscore.py` (the scoring functions — USE THEM, do not re-derive), and
`ml/rollout_spatial.py` (the evaluator you are modifying: `roll_step` ~:747,
the head loop ~:2180-2400, the long/future roll ~:2543 region, `read_sv`
:1430ish, results writing via `write_results`).

## The prime directive

**A deterministic head's evaluation must be BYTE-IDENTICAL to today's.** The
gate head (`e017_u1_s0`, pinned corridor 0.643) and every archived number run
through the exact code path they always did; every change below activates
ONLY when a head's checkpoint `args` carry `fgn_eps > 0`. `read_sv`,
`GATE_REF`, the scopes, `skill_block`, `accumulate`, the audit block and the
E-055 unpooled read-out are untouched for deterministic heads. New keys are
written BESIDE existing ones, never instead.

## What an FGN head's evaluation IS

FGN's own rollout convention (verified against google-deepmind/weathernext,
`weathernext2/fgn.py`: noise is drawn inside every predictor call, so an AR
rollout resamples it each step; the model seed is fixed per trajectory):

- **M member trajectories per start** (`--ens-members`, default 8, min 2).
  Member m has its own CPU `torch.Generator` seeded
  `stable_hash(a.ens_seed, m)` (`a.ens_seed` default 0; use a simple
  documented formula, e.g. `(ens_seed * 1000003 + 59 + m)`); at each rolled
  step it draws ONE ε [k] which is **shared across all pixels** of that
  member's step (expand inside `roll_step`) — ε is global per (member, step),
  exactly FGN's "one draw for the whole field". Draw on CPU, move to device
  (device-independent streams, same reason as temporal.py's eps_gen).
- Members are rolled SEQUENTIALLY (memory: one Zwin at a time). Per start,
  keep each member's decoded fields per horizon in RAM as float16
  [M, Hh_eff, P, C] (~80 MB per member per 12-month monthly start — fine),
  then reduce per h once all members are done.
- `roll_step` gains an optional `eps` argument ([k] tensor or None); when not
  None it is expanded to each chunk's rows and passed to the model. When the
  model was built with eps_dim>0 and eps is None, the model itself raises —
  that guard stays.

## The read-outs (all NEW keys; use ml/probscore functions on numpy)

Per scope (same scope masks as today), per horizon h, accumulated over
starts exactly the way `accumulate`/`skill_block` do (streaming sums, obs
masks intersected the same way — read those functions and mirror their NaN /
obs-mask discipline):

1. **Ensemble-mean skill** — the ensemble-mean field goes through the
   EXISTING `accumulate`/`skill_block` machinery so the entry's scope blocks
   (corridor AUC etc.) are computed by the identical code as every archived
   number, just fed the mean field. This is E-057 F1's comparison
   (ensemble-mean corridor AUC vs the znoise pair 0.7235) and it must be
   mechanically comparable. `entry["meta"]["fgn"] = {"members": M,
   "mode": "ensemble_mean", "eps_dim": k, "ens_seed": ...}` so no reader can
   mistake it for a single-forward number.
2. **`ens_prob` block per scope** (new key beside the scope blocks):
   per-h `crps` (fair, `probscore.crps_ensemble` on the member fields vs
   truth, masked to the scope ∩ obs — NOTE the weathernext NaN rule:
   member values are NaN'd where truth is unobserved so spread can't hide
   there; probscore's crps_ensemble already masks on obs, use it),
   `spread`, `rmse`, `spread_ratio` (`probscore.spread_error`),
   `mse_mean`/`mean_var`/`mse_sample` (`probscore.ensemble_decomposition`),
   plus `crps_auc` = mean over h of (1 - crps/crps_clim) where crps_clim is
   the CRPS of the M-member CLIMATOLOGY ensemble... **NO — do not invent a
   reference without a registered definition.** Ship the raw per-h crps /
   spread / decomposition and a plain `crps_mean` over horizons; reference
   choices are an analysis-time decision and EXPERIMENTS.md's, not this
   diff's.
3. **Transport**: per member, per h, `read_sv(zhat_member)` (the pooled
   legacy read; unchanged function). The legacy `probe_pts` gets the
   ENSEMBLE-MEAN-FIELD read (so `amoc_bands` keeps its meaning as "bands of
   the point forecast"); a new `probe_pts_ens` records all M member reads,
   and the results gain `amoc_bands_ens`: per band, the fair CRPS of the M
   member transport values vs truth (`probscore.crps_ensemble` on the [M, n]
   series), spread–error, and — the dip instrument — `probscore.brier_dip`
   at threshold (deseasonalised transport < −1σ of the training-year target
   series; compute σ from the same deseasonalised series the bands use, and
   RECORD the threshold value in the output). Same for the unpooled read
   (`read_sv_unpooled`) when present, under `amoc_bands_ens_unpooled`.
4. **The dispersion battery (E-057.3's instrument), on the long + future
   rolls**: for an fgn head the long hindcast and future roll also run M
   members (ε per step per member). Do NOT store M full trajectories; per
   step, record the member SPREAD of `read_sv` (transport) and the field
   member-variance mean over the corridor — two scalars per step per roll —
   as `long_dispersion` / `future_dispersion` arrays. Genuine dynamics:
   dispersion grows with lead; replay: flat. (The multi-context-end battery
   itself stays a dispatch-time protocol — this diff just makes each roll
   emit the dispersion curve.)
5. **Trajectory dump** (`--dump-roll`): for fgn heads dump member 0 only,
   with `"member": 0` in the dump meta (the animation stays one trajectory;
   M× dump volume is not worth it).

## Construction / refusal changes

- Where the evaluator builds `TemporalTransformer(...)` from `ta`, pass
  `eps_dim=int(ta.get("fgn_eps", 0) or 0)`. A checkpoint with fgn_eps>0 and
  `--ens-members < 2` refuses at argv time. Print a banner naming M, k,
  ens_seed and the per-(member,step) shared-ε convention.
- `label` for an fgn head gains a token (e.g. `fgnM8`) so no fgn entry can
  collide with or be mistaken for a deterministic one.
- Cost guard, stated not silent: print the M× step multiplier and the
  planned step count at head start (the existing `n_skill + ...` line ×M).

## Tests — `tests/test_fgn_roll.py` (new; CPU; plain python asserts)

Build a TINY synthetic world (mirror whatever existing rollout toy harness
exists — check tests/ for rollout tests first and reuse their fixtures; if
none is reusable, build the minimal npz/codec/head fixtures the way
tests/test_fgn_head.py's end-to-end test does):

1. **Deterministic-path purity**: run the evaluator (or the narrowest
   callable slice of it that exercises roll_step→accumulate→skill_block→
   write_results) on a deterministic toy head with the patched code and
   assert the results dict is IDENTICAL (deep compare) to the pristine
   tree's output on the same inputs. If running the full main() twice is
   impractical, achieve the same guarantee by asserting (a) roll_step with
   eps=None is bitwise-unchanged vs a pre-patch copy of its logic on random
   tensors, and (b) no new keys appear for a det head end-to-end.
2. **Member determinism**: same ens_seed ⇒ bitwise-identical member
   trajectories and identical ens_prob numbers across two runs; different
   ens_seed ⇒ different members.
3. **Shared-ε within a member-step**: instrument that all pixels of one
   member-step saw the same ε (e.g. two chunks of the same step produce
   outputs consistent with one ε; assert via a hook or by chunk-size
   invariance: chunk=P vs chunk=P//3 give bitwise-identical member fields).
4. **CRPS wiring**: the ens_prob crps of a toy where members == truth
   exactly is 0; M=identical-members reduces to the MAE identity; numbers
   match a direct probscore call on the stored member fields.
5. **M=1 refusal**; fgn head + old evaluator args behave: a det head with
   `--ens-members` set is a no-op (flag read only for fgn heads).
6. **Dispersion arrays**: on a toy fgn head, long-roll dispersion arrays
   have the right length and are ≥ 0; a zero-film head (ε ignored at init)
   yields dispersion ≈ 0 — which is itself the collapse signature the
   instrument must be able to show.

## What NOT to touch

`read_sv` internals, `GATE_REF`/`GATE_TOL`, `corridor_pixels`, `accumulate`,
`skill_block`, `write_results`' format for existing keys, `--export-mask`,
the E-047 block paths, `ml-train.yml`. No behaviour change of any kind for
deterministic heads is the acceptance bar.

## Definition of done

`python3 tests/test_fgn_roll.py` green; `python3 tests/test_fgn_head.py`
still green; any existing rollout-related CPU test still green (list what
you ran); a readable diff; NO commit (the main session commits). Report:
files changed + line counts, verbatim test outputs, how test 1's purity
guarantee was achieved, memory estimate of the member buffer at monthly
xl144 M=8, and any deviation from this spec with its reason.

# E-080 handover — the cut-off mirrored double cone (stage-1 hourglass design)

**Written 2026-09-16** by the Fable session that produced E-080 revisions 1–3, for
any agent picking this up. E-080 is a **design and a pre-registration** for the
next shape of the stage-1 codec — the encoder that turns one place at one time
into an embedding. **Nothing is implemented, measured, dispatched or rented.**
Read this, then the deck's notes PDF, then the plan; the plan is the deck's
source and is authoritative where they differ.

- [E-080 plan — every slide's text and speaker notes](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E080_hourglass_cone.md)
- [Slides with speaker notes, PDF](https://blauewelt.github.io/earth/ml/plans/E080_hourglass_cone_deck_with_notes.pdf)
- [Slides only, PDF](https://blauewelt.github.io/earth/ml/plans/E080_hourglass_cone_deck.pdf)
- [PowerPoint, notes in the notes pane](https://github.com/blauewelt/earth/raw/main/ml/plans/E080_hourglass_cone_deck.pptx)
- [The deck on one slide, with an empty panel reserved for the results figure — PDF](https://blauewelt.github.io/earth/ml/plans/E080_hourglass_cone_summary.pdf) (also slide 21 of the deck; it carries its own two summary-sized pictures, `E080_figures/summary_hourglass.png` and `summary_cone.png`; the .pptx sits beside it — when E-080's numbers exist, the figure goes into that panel)
- [The hourglass at two scales — stage 1 over raw values, stage 2 over embeddings, one slide, PDF](https://blauewelt.github.io/earth/ml/plans/E080_hourglass_two_scales.pdf) (also slide 22 of the deck; stage 2's mirrored future cone is design intent, not part of the pre-registered experiment; a white-background twin sits beside it as `E080_hourglass_two_scales_light.pdf/.pptx`)
- [The design paper — the current `ml/paper/paper.pdf` (18 Sep 2026) is a design paper for the global ocean–land–atmosphere model built around this hourglass; the North Atlantic results report it grew out of is archived under `ml/paper/archive/v8-2026-09-18-north-atlantic-results/`](https://github.com/blauewelt/earth/blob/main/ml/paper/paper.pdf)
- [The archived report's design section — §4 "A cone-based design for the next codec and forecaster", light PDF](https://github.com/blauewelt/earth/blob/main/ml/paper/archive/v8-2026-09-18-north-atlantic-results/paper.pdf) (and the [dark build](https://github.com/blauewelt/earth/blob/main/ml/paper/archive/v8-2026-09-18-north-atlantic-results/paper_dark.pdf); the hourglass carried into the paper as a full specification of both stages — inputs and tokens, geometry, tasks, model, the decoupled Gaussian objective, the stability guards, the forecaster as a residual on a contractive linear propagator with direct multi-lead targets, and the ordered comparisons that decide each choice; its Figure 6 is the two-scales drawing under `THEME_PAPER_*` in `E080_build/figures.py`, written by `ml/paper/make_hourglass_fig.py`)
- [E-080 entry in the experiment log](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-080)
- [E-069 · the cone codec as built (what E-080 replaces)](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E069_cone_codec.md)
- [E-071 · cone v2 (the geometry E-080 builds on)](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E071_cone_v2.md)

## 1 · The design in one paragraph

The input cone loses its tip: the present is a **waist** of ~13 cells (gradients
need neighbours), read and reconstructed, and the embedding is made there. A
**prediction cone** — the past cone point-mirrored through the anchor, +1…+6
pentads — is sampled the same way and used only as targets. Training draws one of
**four tasks** per example (forecast from waist + past; forecast from the waist
alone, the built-in snapshot twin; nowcast the hidden present from the past under
five dropout patterns; fill-in), under E-069b's rules (knowable targets only, no
copy targets, every loss family against its own predict-the-mean bar). The cone's
**shape is learnable per region**: a fixed 24-point sunflower warped by eight
numbers per 1° cell and channel group — a drift **d** toward the source, a waist
ellipse Σ₀ and a velocity-spread ellipse Σ_v (the cone at lag ℓ is Σ₀ + ℓ²Σ_v),
the ellipses stored as **matrix logarithms** so nothing is in kilometres and no
angle wraps. The **direction is shared per flow, the scale is per channel**: each
channel gets one log-scale offset s_c and reads the group's shared per-location
tokens through its own Gaussian **aperture** weight, at zero extra tokens. The
geometry learns **gate-first** (soft aperture over cone v2's fixed candidate dots,
so the data loader's reads never depend on live weights) and is then **hardened**
(dot tables re-baked from the learned ellipses at epoch boundaries). Guards: drift
capped at cone v2's design speed, a coarse-10°-plus-1°-residual prior, mirror tie,
stop-gradient on target positions, frozen evaluation geometry, the far ring kept
and never gated. Pre-registered: at the RAPID line (the 26.5° N mooring array that
measures the overturning) the learned surface drift should point south-west, up
the Florida Current; the model's attention on deep Argo profiles should sit north,
up the Deep Western Boundary Current. Three arms (fixed / learned / learned-primed
from climatology) × three seeds, verdict rule written before any run.

## 2 · Decisions and who made them (so nothing gets re-litigated)

| when (2026) | decision | by |
|---|---|---|
| 15 Sep a.m. | the four points above (waist, prediction cone, task mix, learnable per-area cone) | Chris |
| 15 Sep | 1° parameter maps (not 2°, not 10°) with a coarse+residual prior; **no Argo ellipses** — Argo is the nearest-profile tokens of E-071 §3 / E-076, nothing elliptic to learn; depth-dependent sourcing is an attention diagnostic (R4) | Chris, after a second agent's review (comments A) |
| 15 Sep | matrix-log ellipses, all geometry parameters dimensionless (reviewer comments C, D — accepted; the mechanism is Adam's unit-sized step, not gradient magnitude) | Fable, Chris agreed |
| 15 Sep | direction per flow, scale per channel, per-channel aperture over shared tokens (Chris's question "don't we need a scale for each channel?") | Chris + Fable |
| 15 Sep | gate-first then harden instead of a deformable gather at continuous positions (reviewer comment B — accepted on the systems argument; its quoted numbers are not from our record) | Fable, Chris agreed |
| 15 Sep | reviewer comment E (drop the far ring for a 1,400 km bound) **not adopted**: it reverses E-071 §4 and would exclude the equatorial Kelvin wave (2.8 m/s ≈ 7,000 km in 30 days); ring-vs-no-ring is the first ablation after the three arms | Fable; Chris has not overruled |
| 16 Sep | time is NOT sunflower-sampled (dense pentads inside the 30-day window, by design — the codec owns the last 30 days, stage 2 the long window); two cheap additions agreed: golden-angle phase rotation per lag, and a per-channel **time aperture** τ_c (learned memory) initialised from `FAMILIES.tau_days` | Chris ("go ahead") — being folded into the deck as revision 3.1 |

Facts checked against the repo, for the record: there is **no "Family 9"** (families
are 3–8 and 10/10.1); no run in the log shows an aperture "freezing at 850 km"; the
"28–40 steps/s" and "0.045 ms gate" figures a reviewer quoted are not in our record.
Where a reviewer's point was right it was right on its own merits.

## 3 · Open decisions (slide 19 of the deck)

Future horizon +6 vs +1…+2 · task shares 0.35/0.15/0.25/0.25 · prior strength on
the 1° residual (held-out or fixed) · keep the far ring (yes, until the ablation) ·
seasonal drift term (phase 2) · harden interval N = 2 k and geometry LR ×10 (first
guesses). Any of these is one line to change; none affects the build order.

## 4 · Build order (slide 18) — where implementation starts

Five steps, each landing on main with its test before the next: (1) waist and
mirrored future cone in `ml/cone.py`, coverage report extended with the mirror
identity; (2) the four-task plan next to `ml/cone_codec.py::default_plan`, with a
10 k-example test that no target's channel is invisible; (3) the aperture gate in
`ml/cone_codec.py` — an infinite aperture must reproduce the ungated model bit-for-bit,
and autograd must match finite differences on every geometry number; (4) the
parameter maps (matrix-log storage, log-space floors, prior, mirror, stop-gradient,
init = the fixed hourglass) and the harden step in `ml/cone_sampler.py` (per-region
dot table swapped only at epoch boundaries); (5) read-outs in `ml/train_cone.py`
and three recipes `f7l2-hourglass-{fixed,learned,primed}`. Per `ml/CLAUDE.md` §0b
the implementation is Opus-subagent work to this spec; the main session plans,
reads diffs and results. Do not dispatch anything without Chris's word; budget ~2–3
4090-hours per seed on family 7.2, ~25 for the nine runs.

## 5 · How the deck is rebuilt, and the conventions it must keep

The build lives in `ml/plans/E080_build/` (pptxgenjs deck builder, a `pPr`
post-fix without which LibreOffice drops bullets, the figure scripts, the
notes-PDF script, `rebuild.sh` — always rebuild through `rebuild.sh`). The
spec is the plan file: one `## Slide N ·` section per slide, a bold headline, an
optional `[FIGURE: …]` description, bullets, and a `*Notes.*` paragraph that is
copied **verbatim** into the PPTX notes pane and the interleaved notes PDF (slide
page, notes page, …). Style matches the E-069 training-example deck (dark navy,
serif titles, blue/gold accents, footer with deck name and slide number).
Conventions Chris set, all of them standing: no acronym or code name without a
plain-English sentence; every reference another reader would not know is a
**clickable link to its definition** on the slide and is explained in the notes;
markdown links, one per line, in every reply; no choice dialogs — options in prose,
name the pick, proceed reversibly. After a rebuild: verify every rendered page,
counts (N slides / N notes / N + 2N pages, plus the one-page summary export), then commit the plan, the deck
files, the summary export and the figures in one commit, add a revision note to the E-080 entry in
`ml/EXPERIMENTS.md`, keep `docs.html`'s DOCS description current, push with
`node scripts/git_api_push.mjs --branch main --token-file /home/claude/.gh_pat`
(never force), fast-forward `gh-pages` with `force:false`, and verify the served
bytes changed. No `ml/OVERVIEW.md` stamp while nothing is dispatched (`ml/CLAUDE.md` §0g).

## 6 · What not to do

Do not add Argo ellipses back; do not coarsen the maps to 10°; do not replace the
ring with a bound without Chris; do not read at interpolated positions in the
loader; do not let the learned aperture touch the loss weights or the evaluation
geometry; do not quote a reviewer's numbers as ours without an artefact.

## 7 · Changes required by the 22 September paper review (before any build step)

The review (`PAPER_REVIEW_2026-09-22_feedback.md`) and its response
(`PAPER_REVIEW_2026-09-22_response.md`) changed the design paper; the build must
follow the paper, and four items in this handover are superseded by it:

1. **Targets are fixed in every phase.** Target queries (mirrored future cone,
   hidden waist, fill-in cells and points) are drawn from the fixed hourglass of
   the initial table and never re-baked from the learned ellipses; the re-bake
   moves the *input* table only. R5 (the target-variance flag) is dropped; the
   mirror of the learned cone is a later, separately guarded comparison.
2. **The decoder-reads-latents path is closed at weight zero**, not 0.25
   (`cone_codec.py` `aux_latent_w`, `train_cone.py:149`).
3. **The admissibility certificate is deterministic**: enumerate the boundary
   cases per anchor bin (earliest input lag, latest target lead, latest point
   observation time) against the period boundaries of paper §3.5; the random
   4,096-anchor draw stays as a diagnostic only. The cone trainer also needs the
   period split of §3.5 (training 1982–2020 less 2009 and 2017; 2021–2024
   retrospective; 2025– prospective) and the `available_at` rule for as-issued
   runs.
4. **The forecasting benchmark, paper comparison (0), runs before the geometry
   comparison (2)**; A0 is trained first as its fixed codec, and A1/A1g/A2 are
   dispatched only if (0) locates a limitation geometry can address.
   Comparison (2) is at equal token budget and adds a coarse-global-summary
   arm in place of the ring.

Also from the paper: `z` is deterministic (no log-variance head on the latent),
the growth ellipse is capped at the design reach and not floored at the design
speed, and the source contract fields (`support_start`, `support_end`,
`available_at`, `product_version`, `lineage`) are added to the registries
before a run reads a store under the as-issued mode.

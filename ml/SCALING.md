# Scaling audit — do we have enough data for the models we train?

All counts measured by `scaling_audit.py` (2026-08-06); re-run it after
changing channels, models, or splits. The one-line summary: **every stage
of this pipeline is data-limited, not model- or compute-limited — the
lever is more observed months and channels, not bigger networks or longer
training.**

## The inventory

| quantity | measured |
|---|---|
| observed values in the 12-channel tensor | 18.1 M |
| pixel-months (401 months × 5,787 ocean px) | 2.32 M |
| stage-2 transitions (pixel-month → next) | 2.31 M |
| RAPID truth months (total / held-out) | 240 / 36 |
| PixelMAE codec (any d_z 8–64) | ~0.90–0.92 M params |
| stage-2 mid (d96×3) / large (d192×4) | 0.35 M / 1.81 M params |
| mini probe transformer (d64×2) | 0.11 M params |
| linear probe / K=24 concat probe | 33 / 769 params |

## Chinchilla-style read, stage by stage

The Chinchilla rule of thumb for LLMs — optimal params ≈ tokens/20 —
transfers only loosely, but as an order-of-magnitude anchor:

**Codec.** 18.1 M observed values ÷ 20 ≈ **0.9 M params — exactly the
codec's size**, by coincidence rather than design. But two corrections
both point the same way. First, epochs: the 30k-step runner pass sees
6.6 epochs of the corpus, and data-constrained scaling work (Muennighoff
et al. 2023) finds repetition worth close to fresh data only up to ~4
epochs — so the 30k-step budget already exhausts the useful-repetition
zone, which matches observation (5× more steps bought +4 pts of chan% in
the 4-channel pilot and nothing on the probe; trainprobe curves are
flat). Second, independence: 1° ocean cells are spatially correlated over
several hundred km and ~2–3 months, so effective sample count is
several-fold smaller than the raw 18 M. Verdict: **the codec is at or
past the data-optimal size. Do not grow it; feed it.** Each new dynamic
channel with a long record adds real tokens: ERA5 wind stress (monthly
1940→, ~2.4× the months of GLORYS), DUACS SLA (1993→), OISST monthly
(1982→). Adding channels also lengthens the RG-poor early period's
usable signal.

**d_z (the sweep in flight).** Note from the table: d_z barely changes
parameter count (894k → 916k from d_z=8 to 64) — the sweep measures how
much *information the bottleneck transmits*, not model capacity. That is
the right question at fixed data.

**Stage-2 temporal transformer.** 2.31 M transitions; token-anchored
optimum ≈ 0.1 M params. The mid config (0.35 M) is already at/above it;
large (1.81 M) is ~15× over. The sweep-so-far confirms the prediction the
arithmetic makes:

| config | params | chan% (seeds) | r_tmp (seeds) |
|---|---|---|---|
| small d64×2 K12 | 0.11 M | 30.1 (30.7/29.6) | 0.29 (0.27/0.31) |
| mid d96×3 K24 | 0.35 M | 30.9 (31.4/30.4) | 0.32 (0.31/0.33) |
| mid K36 | 0.35 M | 31.0 (31.5/30.4) | 0.21 (0.33/0.09) |
| mid K6 | 0.35 M | 30.2 (30.7/29.7) | 0.26 (0.33/0.19) |
| large d192×4 K24 | 1.81 M | *(running)* | |

Capacity and context length move chan% by ≤1 pt across a 16× parameter
range; the probe column wobbles exactly as its ±0.57 CI (METRICS.md)
says it must. **The downstream transformer is not the bottleneck.** The
honest "more realistic" upgrade is not a bigger transformer but a bigger
*task*: multi-horizon prediction (t+3, t+6) and cross-pixel attention
(spatial context), both of which add supervision per parameter rather
than parameters per supervision.

**Mini training-time probe.** 0.11 M params, deliberately under the
optimum — it is a measurement instrument, and the sweep shows it tracks
the larger configs' chan% within ~1 pt. Faithful enough; keep it small.

**The probe task (the hard wall).** Ridge with 33 params on 204 train
months is fine; the K-concat probe at 769 params exceeds its 190 train
points (p > n) — its instability is arithmetic, not mystery. And no model
choice fixes the TEST side: 36 autocorrelated months ≈ 15 independent
samples. The probe's data ceiling is the project's binding constraint,
and only new truth (Florida cable 1982→, OSNAP/MOVE/SAMBA) or the
year-blocked k-fold protocol (METRICS.md) raises it.

## Ranked levers (what to spend effort on)

1. **More truth for the probe** — Florida Current cable, OSNAP, MOVE,
   SAMBA; year-blocked k-fold protocol. Raises the ceiling on the
   question the project exists to answer.
2. **More channels with long records** — ERA5 τ, DUACS SLA, OISST
   monthly: real new tokens for the codec, more months of usable signal.
3. **Richer supervision per parameter** — multi-horizon and spatial
   stage-2 objectives.
4. ~~Bigger models~~ — the arithmetic and the sweep both say no.
5. ~~Longer training~~ — past the ~4-epoch useful-repetition zone
   already.

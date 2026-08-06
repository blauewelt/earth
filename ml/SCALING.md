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

**d_z — answered 2026-08-06 evening.** Four codecs at matched 30k steps:
chan% 28.6 / 30.3 / 30.5 / 30.6 for d_z 8/16/32/64. The bottleneck
saturates between 16 and 32; only 8 pays (−2 pts). d_z was never a
capacity axis (params barely move, 894k→916k) — it measures transmitted
information, and 32 dims already carry everything the current 12
channels supply. Keep 32; revisit only after the channel count grows.

**Stage-2 temporal transformer — corrected 2026-08-06 evening.** The
first draft of this section predicted, from a transitions-as-tokens
anchor (2.31 M transitions -> ~0.1 M params optimal), that the large
config would not beat mid. **The experiment said otherwise**, and a
steps-matched control separated the confound:

| config | params | steps | chan% (seeds) |
|---|---|---|---|
| small d64×2 | 0.11 M | 2000 | 30.1 (30.7/29.6) |
| mid d96×3 | 0.35 M | 2000 | 30.9 (31.4/30.4) |
| mid d96×3 | 0.35 M | 4000 | 31.9 (32.4/31.4) |
| large d192×4 | 1.81 M | 4000 | 35.9 (35.5/36.2) |
| xlarge d256×5 | 3.98 M | 4000 | **37.7** (37.6/37.7) |

Doubling steps bought mid +1.0 pt; at matched 4000 steps, capacity buys
**+4.0 pts** (seeds two points apart — decision-grade by METRICS.md).
The K variants (6/36) at mid size stayed flat, so context length is not
the axis; width×depth is. Where the arithmetic went wrong: a stage-2
"token" is not a transition-count, it is a 32-dimensional continuous
vector — counted in VALUES, D ≈ 74 M -> anchor ≈ 3.7 M params, and large
(1.81 M) is still under it. Lesson recorded: for continuous multivariate
sequences, anchor on value-count, not token-count — and run the control
before believing either. xlarge (d256×5, 3.98 M — at the
value-count anchor) confirmed the curve with the tightest seeds of the
whole sweep: 37.6/37.7. The gain per parameter-doubling is decelerating
(+4.0 for mid→large at ×5, +1.8 for large→xlarge at ×2.2) — roughly
log-linear and approaching the anchor as theory would like. Next rung:
d320×6 (7.44 M) on runners, past the anchor, to find the turn. Probe r
stayed inside its noise band throughout (large 0.27/0.39; xlarge
0.25/0.46), as METRICS.md demands it must.

**Mini training-time probe.** 0.11 M params, deliberately small — a
measurement instrument. It ranks codecs faithfully (that is its job),
but note it UNDERESTIMATES achievable dynamics: the large stage-2 gets
+5 pts over what the mini reads on the same embeddings. Use it for
curves and rankings, never as the ceiling.

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
3. **Scale stage 2** — the steps-matched control shows capacity buys
   real dynamics (+4 pts to d192×4); headroom to ~3–4 M params by the
   value-count anchor. Runner-sized job, controls mandatory.
4. **Richer supervision per parameter** — multi-horizon and spatial
   stage-2 objectives; compose with (3).
5. ~~Bigger codec~~ — at/past its data anchor until new channels land.
6. ~~Longer codec training~~ — past the useful-repetition zone.

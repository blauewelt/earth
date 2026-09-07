# E-076 · Family 8 — the measurement is never missing, only far away

**Written 2026-09-07**, from Chris's proposal of the same day: *"In our
design, in a given cone, either there is data or there isn't. But there's an
argument to be made for always having the 3 (or 5 or 7) nearest points in a
cone (even for very sparse data, for example Argo floats). That way, a
channel is never empty, the measurement is just very far away."* The deck
version is slides 55–56 of the representation survey; this is the plan with
the numbers, the rules, the storage design, and the one ablation that decides
whether family 8 is built.

This is a PLAN. Nothing is dispatched, nothing is downloaded, no box is
rented. Every number is measured in this programme and cited, taken from a
named source, or arithmetic shown in place.

Read with:
[E-070 · the family-7 build spec](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E070_family7_build.md)
(what family 8 keeps unchanged),
[E-071 · cone v2](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E071_cone_v2.md)
(§3–4: the profile tokens this plan takes to their conclusion),
[the family-7 data handover](https://blauewelt.github.io/earth/docs.html?f=docs/FAMILY7_DATA_HANDOVER.md)
(§5, what "missing" means today, channel by channel),
[the survey deck, slides 55–56](https://github.com/blauewelt/earth/blob/main/ml/figures/geofm_survey/geospatial-representation-models-with-notes.pdf).

---

## 0 · The one-paragraph answer

Keep family 7's dense groups exactly as they are. For every SPARSE channel —
today the Argo interior, tomorrow moorings, altimeter tracks, ship
measurements — stop asking "is there a value at this cell and bin" and
instead give the cone the **k nearest observations, however far away, each
carrying its own distance**: the token is `(value, Δx, Δy, Δt, n_R)`, the
value, its offset east and north in kilometres, how long ago in days, and
how many observations lay within the search radius. Three rules make it
safe: one-sided in time (only observations at or before the anchor's pentad,
or the forecaster leaks), bounded (a radius `R_max` and a window `T_max` per
channel, beyond which the miss token survives but becomes rare and true),
and `k` chosen from the data so that the k-th neighbour typically sits at one
correlation length. For Argo that arithmetic gives **k = 5 within 30 days**.
Family 8 is therefore a hybrid — sunflower for dense channels, nearest
observations for sparse ones — and it is no longer a single tensor: the dense
groups plus an observation store and a neighbour index. That is its cost and
its prize, because it is the door to raw observations without gridding them
first. One ablation on family 7, about $1, decides whether it is built.

---

## 1 · What is wrong with "present or absent"

Family 7's Argo group is written into 252 of 3,142 pentads — **8 %** — and
at every other bin every one of its 32 channels is a miss token
([handover §5](https://blauewelt.github.io/earth/docs.html?f=docs/FAMILY7_DATA_HANDOVER.md)).
On the North Atlantic mapping the same channels were 83–97 % missing tokens
(E-052 addendum). The group is 80 % of the tensor's bytes. And the signal is
real: **#438 (E-044c arm A3 — the pentad head with the Argo-target windows
excluded)** made the forecast *worse*, 0.570 against ~0.50, when those targets
were removed
([E-044c](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-044c)).
So the interior carries forecastable information, and the representation
throws most of it away by asking a question — "is there a value here, now" —
that a sparse observing system almost always answers "no".

The question the observing system *can* answer is "what is the nearest
measurement, and how far away is it". A float profile 200 km away and three
days old is not nothing; it is a measurement with a known displacement, and
the displacement is information about how much to trust it.

---

## 2 · The construction

### 2.1 · The token

For a sparse channel `c` at anchor `(y, x, b)`:

```
tokens_c = the k observations of c nearest to the anchor in space-time,
           each as  (value, Δx_km, Δy_km, Δt_days, n_R)
```

- `value` — the observation, standardised and anomalised exactly as the
  dense channels are (§2.5).
- `Δx_km, Δy_km` — the observation's offset from the anchor, east and north,
  great-circle, signed. Kilometres rather than degrees so the model does not
  have to learn the cosine of latitude.
- `Δt_days` — how long BEFORE the anchor's pentad the observation was made,
  `≥ 0` always (§2.2).
- `n_R` — how many observations of `c` lay within `(R_max, T_max)` of the
  anchor. The local data density; the honest proxy for how much the k
  neighbours can be trusted, and the feature that stops a Southern Ocean
  anchor with two floats looking like a North Atlantic anchor with forty.

"Nearest" is measured by a space-time metric `d² = (Δx² + Δy²)/L² + Δt²/T²`
with `L` the channel's correlation length and `T` its decorrelation time —
the same two numbers the cone families already carry.

### 2.2 · The three rules

**One-sided in time.** Only observations at or before the anchor's pentad.
An observation from the next pentad is the future, and a forecaster that has
seen it is not a forecaster. This is a hard constraint on the search, not a
mask applied afterwards.

**Bounded.** A search radius `R_max` and a window `T_max` per channel. Inside
them the k nearest are taken; if fewer than k exist, the remaining slots carry
the miss token — the same token family 7 uses at 92 % of bins, made rare.
Without the bound an unbounded search returns a profile from another basin
or another decade and asks the model to learn that it is worthless, which it
would, slowly, at the cost of tokens.

**k from the data.** Choose k per channel so that the k-th neighbour
typically sits at one correlation length. A smaller k under-uses the
observing system; a larger one pads the cone with neighbours the anchor is
already decorrelated from.

### 2.3 · The arithmetic for Argo

Roughly 4,000 active floats on a 10-day cycle
([Argo](https://argo.ucsd.edu/)) give about 400 profiles a day, **≈ 2,000
per pentad worldwide**. The ocean is 3.61 × 10⁸ km², so one profile per
≈ 1.8 × 10⁵ km²; for a Poisson field the mean nearest-neighbour distance is
0.5/√λ ≈ **210 km in a single pentad**. With `T_max = 30` days the search sees
six pentads of profiles, 12,000, one per 3.0 × 10⁴ km²: nearest ≈ **87 km**,
and the k-th neighbour at ≈ √k times that, so the fifth sits at ≈ **195 km**
— one surface correlation length. Hence **k = 5, T_max = 30 d**, and
`R_max = 1,000 km` as a generous bound that is rarely binding.

Before 2004 the search finds nothing (the float array reached 3,000 floats in
2007). Then "none within R" is the honest, measured answer — a property of
the observing system's history, not a gap to be papered over — and the model
learns from `n_R = 0` that the interior was unobserved, which is true.

### 2.4 · What is NOT changed — the hybrid

Dense gridded channels — currents, sea-surface height, mixed-layer depth,
sea-surface temperature, sea ice, and the fifteen reanalysis channels — keep
the sunflower and the 3 × 3 patch exactly as in family 7. For a dense channel
the k nearest observations are always the immediate neighbours, and the
sunflower already encodes structure at several scales that a k-nearest set
would collapse to the 3 × 3. Family 8 changes how a SPARSE channel enters
the cone and nothing else about the geometry.

### 2.5 · Anomaly and normalisation

The per-channel day-of-year harmonic climatology of E-071 §2 is evaluated at
the observation's own `(lat, lon, day)`, not at the anchor's — the anomaly
belongs to where the measurement was made. Standardisation constants are per
channel as today. Both are computed on training years only, as the frozen
protocol requires.

### 2.6 · Granularity — the footprint of a measurement, not only its distance

Chris, 2026-09-07: *"Can we augment the family 8 proposal to not only keep
distance but also granularity for each channel? … eg for a 1 degree dot that
exists also for the 0.25 dot contained in it."* And: *"maybe there are two
options here: a) model coarseness as a flat 'granularity' information, b)
model coarseness as the distance to the center of the coarse dot."*

The case that motivates it is already in family 7. A 1° reanalysis value
(`g100`) is served to all sixteen 0.25° cells inside it, and to the nine cells
of a lag-0 patch, as if it were nine separate measurements that happen to
agree. It is one measurement. The model cannot tell "nine cells read 0.31"
from "one cell was read once and copied", and those are different amounts of
evidence.

**Take: do both, because they are different quantities.** Option (b) is a
statement about WHERE the measurement is; option (a) is a statement about
WHAT AREA (and what time span) it averaged. Neither can be recovered from the
other, and each is nearly free.

**(b) is almost already there once a gridded value is an observation.** Treat
a 1° cell's value as an observation located at the cell's centre. Then
`(Δx_km, Δy_km)` of §2.1 carry the "distance to the centre of the coarse dot"
for free — 0 to 78 km for a 1° cell, 0 to 20 km for a 0.25° cell — and the
nine patch cells point at the SAME centre from nine different offsets instead
of pretending to be nine values. No new field; the token's existing offsets
just stop being zero for coarse channels. The costless correction to family
7's "served as the same cell" lie.

**(a) carries what (b) cannot: the support.** Two measurements at the same
offset can still differ in what they averaged over — a point profile, a
0.25° satellite cell, a 1° storage cell, or NCEP's native T62 Gaussian grid
(≈ 1.9°), which is the honest spatial support of every `g100` channel however
it is stored. The same holds in time: an instantaneous profile, a daily
composite, a pentad mean, a monthly mean (`rg100`). A distance of 30 km from
a 1.9° average and 30 km from a point profile mean different things about the
anchor, and only the footprint says which. So the token gains two fields:

```
tokens_c = (value, Δx_km, Δy_km, Δt_days, n_R, log2_fp, log2_dt)
   log2_fp = log2(footprint_km / 27.83 km)     spatial support, in 0.25°-cell units
   log2_dt = log2(support_days / 5)            temporal support, in pentad units
```

Both are clamped to a small range (say −4 … +4) and, for a channel whose
support never varies, are constants — cheap, and they let one embedding
serve a channel that is later ingested at a finer native resolution without
changing the schema. Values by source: Argo profile `log2_fp ≈ −4` (point),
`log2_dt = log2(0.02) ≈ −5.6 → −4` (instantaneous, clamped); OISST 0.25°
`(0, 0)`; NCEP at T62 `(log2(1.9/0.25) ≈ +2.9, 0)`; a monthly Roemmich–Gilson
field `(+2, log2(30/5) ≈ +2.6)`. Note that NCEP's value is +2.9, not +2: the
1° grid is how the data is stored, not what it resolved.

**Why not (a) alone, or (b) alone.** (a) alone leaves the nine-patch-cells lie
in place — the model still sees nine tokens claiming nine positions. (b) alone
cannot distinguish a 0.25° cell 20 km away from a T62 cell 20 km away, and
that distinction is most of what "coarse" means for a forecast: a coarse value
is a smoother, lower-variance quantity, and the model must know to expect
less high-frequency content from it. AlphaEarth makes the same choice on
the encoder side — each source's decoder is conditioned on the sensor's
own metadata (its resolution among them) rather than resampling every source
to a common grid first (E-075 §8).

**What this costs.** Two extra float16 fields per sparse token; nothing for
the dense sunflower, whose cells all share one footprint per group and can
carry it as a per-group constant. E-076a (§5) gains one more arm: the
footprint fields present vs. zeroed, on the same seed — a single number
answers whether the model uses them.

---

## 3 · Storage — family 8 is not a tensor

| piece | form | size | state |
|---|---|---|---|
| the dense groups `g025`, `g100` | unchanged from family 7 | 51.8 GB | **exists** |
| the observation store | one table per sparse channel family: `(bin, lat, lon, depth_level, value, source_id)`, float16 values, sorted by bin, with a per-bin offset index | Argo 2004–2024 at 16 levels × T/S: ≈ 2.5 M profiles × 32 values ≈ **0.2 GB** | not built |
| the neighbour index | per anchor set and per sparse channel: `k` observation ids + the four offset features, precomputed once with a per-bin KD-tree over `(x, y)` on the sphere and a bin range for `T_max` | for 2,048 anchors × 3,142 bins × k = 5 × 5 features × float16 ≈ 0.3 GB; for every cell, computed lazily inside the sampler instead | not built |
| `rg100` | **retired** in family 8 — the gridded, mapped monthly product is replaced by the raw profiles it was mapped from | −1.05 GB | — |

The sampler contract changes accordingly: `ml/cone_sampler.py` gains a
second gather path that, for sparse channels, reads the neighbour index
rather than the grid. The token schema for dense channels is untouched, so
every existing test on the dense path keeps its digest.

**The raw Argo source.** The Argo Global Data Assembly Centres publish a
global profile index (`ar_index_global_prof.txt`: one line per profile with
date, latitude, longitude, float and file) and per-float NetCDF profile files
with delayed-mode quality flags. Extracting temperature and salinity at the
sixteen Roemmich–Gilson pressure levels from 2004 onward is a one-off pull;
the index alone answers the density arithmetic of §2.3 exactly rather than by
the Poisson estimate, and is the first thing to fetch.

---

## 4 · What it buys, and what it risks

**Buys.** The interior always present; distance and density as learnable
uncertainty; a fixed token count per sparse channel regardless of coverage
(no ragged batches); the removal of a mapping step between the observation
and the model; and a representation into which moorings (RAPID, OSNAP, MOVE,
SAMBA as observation tokens rather than only as labels), altimeter tracks and
ship data drop without gridding — the primitive Aardvark and GraphDOP use to
forecast from raw observations.

**Risks, each with its answer built in.** An unbounded search returns junk
(`R_max`, `T_max`). Basin-to-basin density bias (`n_R`). Dense channels do not
benefit and could be harmed by the wrong gather (the hybrid — they are not
touched). The representation stops being one array, and every loader, the
Hub range-read front end and the fixture machinery assume one (the dense
groups keep that contract; only the sparse path is new). And the honest
unknown: whether the model can actually USE displacement as uncertainty at
7 M parameters and 20 k steps — which is exactly what §5 measures before
anything is built.

---

## 5 · E-076a · The ablation that decides it — **≈ $1, one seed, no new tensor**

*Absolute description:* one arm inside the next cone-codec wave on family 7.
Replace the Argo group's 32 gridded channels by k = 5 raw-profile tokens per
level-set (`T_max` 30 d, `R_max` 1,000 km, one-sided), everything else
identical to the continuous twin. `params` ConeMAE 7.05 M · `stage` encoder ·
`data` family 7 + the Argo profile store · `arch` unchanged · `steps×batch`
20 k × 256 · `resume` none.

- **Control:** the wave's own continuous arm reading `rg100` as today.
- **Read-outs:** held-out per-family loss against the predict-the-mean bar
  (the E-069b decomposition — anchor, future and dot families separately);
  the subsurface probe; and the per-channel record of hidden interior
  channels.
- **Falsifier, pre-registered:** the anchor family's held-out loss on hidden
  interior channels must drop below the twin's. If it does not,
  distance-as-a-feature bought nothing at this size and budget, and family 8
  is not built. If it does, the second arm is k ∈ {3, 7} to check the
  correlation-length rule of §2.2, the third is the same tokens with
  `n_R` withheld, to measure what the density feature alone is worth, and the
  fourth is the two footprint fields of §2.6 zeroed, to measure whether the
  model uses the support of a measurement or only its distance.
- **Cost:** the profile pull (~0.2 GB, one afternoon) plus one training seed,
  ≈ 1.7 h on an RTX 4090 at ≈ $0.33/h.

What it needs first, in order: the Argo index and profile extraction
(§3); the sparse gather path in `ml/cone_sampler.py` with a test that pins
the dense path's digest unchanged; the k-nearest index builder with a test on
a synthetic float field where the true nearest neighbours are known.

---

## 6 · Prior art

Neural processes and their convolutional form condition on irregular
observation sets with their locations, which is this token's ancestry.
Aardvark Weather (Nature 2025) forecasts end to end from raw observations;
GraphDOP (ECMWF) predicts directly from observations without an analysis
step; ESFM carries missingness as a first-class token. None of them is an
ocean-interior model, and none frames the sparse channel as "the k nearest,
with distance" inside an otherwise gridded cone — that combination is this
plan's, and E-071's profile tokens are its direct ancestor here.

- [Argo — the float array](https://argo.ucsd.edu/)
- [Aardvark Weather (Nature 2025)](https://doi.org/10.1038/s41586-025-08897-0)
- [GraphDOP (arXiv 2412.15687)](https://arxiv.org/abs/2412.15687)
- [ESFM (arXiv 2605.00850)](https://arxiv.org/abs/2605.00850)

# E-078 · Family 10: one schema for every granularity — how fine-grained data enters without a common grid

*Design note, 2026-09-12. Written at Chris's request of 2026-09-11: "our move to
more fine-grained data should actually use a design that can accommodate
channels of varying granularity." It joins two pieces that already exist —
the footprint token of E-076 §2.6 (7 Sep) and the cone-native codec with a
local codec underneath from the embedding-inputs proposal §4 (2 Sep) — into
one storage-and-sampling contract, and says what is built first.*

## 0 · The sentence

**Nothing is resampled to a common grid at storage time. Every source is
stored at its own resolution and cadence in one of three tiers; the sampler
turns all of them into the same token — a value with its offset from the
anchor and its footprint — at read time; and the decoder is queried at a
location AND a footprint, so the model can predict at any granularity it was
ever shown.**

Family 7.1's box-averaged colour (E-077) is the *old* way, done deliberately
to get a channel onto the grid today. This note is the new way, and it is
designed so that the old way is a special case of it (a tier-G group), not a
thing to be thrown away.

## 1 · Why a common grid is the wrong contract

Three facts the programme has already measured:

1. **A coarse value copied into fine cells is a lie the model cannot see
   through.** Family 7 serves a 1° NCEP value (really a 1.9° Gaussian cell) to
   sixteen 0.25° cells and to the nine cells of a lag-0 patch as if they were
   nine measurements that happen to agree (E-076 §2.6). The model cannot tell
   nine readings of 0.31 from one reading copied nine times, and those are
   different amounts of evidence.
2. **A fine value averaged into a coarse cell throws away exactly the part
   the finer source was bought for.** E-077 keeps 1 of 36 pixels' worth of
   information per 0.25° cell and adds `chl_cov` to say how much it threw
   away. Kilometre-scale radiometry is $3.5 \times 10^{14}$ values (family-7
   design note, the data ladder); "it cannot be a tensor" is true, and
   "therefore average it" is the wrong conclusion.
3. **Missing is not damage, and neither is "far away".** Family 8's
   measurement: optimal interpolation from the same five nearest Argo
   profiles beats climatology by 21 % where the gridded product gave the
   codec 3 % (E-076b). Distance is a feature; a grid cell hides it.

So the contract cannot be "everything on a 0.25° pentad grid". It has to be
"everything at its own granularity, and the *token* says what that
granularity is."

## 2 · The token — one schema for all tiers

From E-076 §2.6, restated as the programme's standing schema:

```
token = ( value[C], mask[C],            # the measurement, standardised; mask 1 where measured
          Δx_km, Δy_km, Δt_days,        # WHERE and WHEN, relative to the anchor
          log2_fp, log2_dt,             # WHAT AREA and WHAT SPAN it averaged (footprint)
          n_R,                          # how many such observations were within reach
          source, channel )             # learned embeddings: sensor/product class, channel id
   log2_fp = log2(footprint_km / 27.83)     # spatial support in 0.25°-cell units
   log2_dt = log2(support_days / 5)         # temporal support in pentad units
```

Concrete values, so the ladder reads as one line: a Sentinel-2 pixel (10 m,
instantaneous) is (−11.4, −4 clamped); an Argo profile (−4, −4); a 300 m OLCI
pixel (−6.5, −4); a 4 km OC-CCI cell, daily (−2.8, −2.3); an OISST 0.25° cell,
pentad (0, 0); a 1° box-average (+2, 0); NCEP at its true T62 support (+2.9,
0); a monthly 1° Argo field (+2, +2.6). The two fields are clamped to ±12 and
are *constants per group* for dense sources — free — and per observation for
sparse ones. Nothing else in the schema depends on the tier the token came
from. That is the whole point: **the encoder never learns which file a value
was read from, only what it is, where it is, and how much it averaged.**

Offsets are the correction to fact 1 above: a 1° cell's value is one token
located at the cell centre, seen from sixteen different (Δx, Δy) by the
sixteen anchors inside it, and the lag-0 patch of a fine anchor holds one such
token, not nine.

## 3 · Three storage tiers, one registry

Sources are stored by their *density*, because density decides what a read
costs, not by their sphere or their science:

| tier | what | layout | reader | exists today |
|---|---|---|---|---|
| **G — gridded dense** (≥ ~4 km, daily to monthly, near-complete coverage) | GLORYS, OISST, NCEP/ERA5, OC-CCI at 4 km, DUACS, GREP levels | one bin-major `.npy` (or zarr) per group at its **native** grid, own `bin_first` / own bin index, z-scored, NaN = unobserved | a slab read (one bin, one group) + a block/nearest lookup at the anchor; footprint = a per-group constant | **yes** — family 7's g025/g100/rg100, family 7.1's oc025 |
| **P — points, profiles, tracks** (sparse in space or time) | Argo (core, Deep, BGC), drifters, moorings, tide gauges, altimeter tracks, SWOT swath samples, GEDI/ICESat-2 footprints, SOCAT | columns sorted by pentad bin with CSR offsets; raw units; per-row (lat, lon, t, footprint, values[C], mask[C]) | k-nearest in a space–time metric, one-sided in time, bounded by (R_max, T_max), miss tokens to a fixed k | **yes** — family 8 (`ml/family8_store.py`) |
| **T — tiles** (fine imagery, ≤ ~1 km, swaths) | Sentinel-1/2, Landsat, NISAR, OLCI/SLSTR at 300 m–1 km, VIIRS, MODIS 1 km | NOT stored as values. A **tile-date catalogue** (footprint polygon, sensor, time, cloud/valid fraction, URL) plus a **token cache**: the local codec's output per tile-date, a few 64-D vectors, stored as a tier-P store whose "values" are the tokens | the tier-P reader — a cached token is an observation with footprint = the tile, `log2_dt` = −4 | **no** — the local codec is the first thing E-078 builds |

Three properties make the tiers one family rather than three projects:

- **One time axis.** Every tier is indexed by the pentad bin from the
  1982-01-01 epoch (tier-P rows keep their exact timestamp inside the bin, so
  Δt is fractional; tier-G carries its own `bin_first`; a monthly group
  carries its own live-bin list as `rg100` does today).
- **One registry.** `family.json` at the family's root lists every group:
  `tier`, grid or store schema, cadence, channel names and units, the
  footprint constants, `bin_first`, file names with sha256, sources and the
  builder commit. Consumers dispatch on `tier`; a new group is a new entry,
  not a new code path. This is `manifest.json` + the npz keys of family 7,
  made explicit and shared by the three tiers.
- **One sampler contract.** `gather(anchor, family) → tokens`: for each
  group, its reader returns tokens in the §2 schema under that group's cone
  family (reach vs lag: A atmosphere, B currents/SSH, C SST/mixed layer, L
  land, and the depth column; E-071 §3–4), with a fixed slot count per group
  so batches stay rectangular. `ml/cone_sampler.py` is already this for
  tier G; `ArgoStore.knearest` is it for tier P; tier T reuses tier P's.

## 4 · The local codec (tier T) — the only new model

A 64 × 64-pixel tile at 10 m is 4,096 pixels × 13 bands; a 0.25° cell holds
~7,700 such tiles. They cannot be dots. The local codec is the programme's
per-pixel masked-autoencoder codec generalised to a tile: input one tile-date
(all bands, the sensor's own metadata — resolution, incidence angle for radar,
sun angle for optical — as conditioning, exactly as AlphaEarth conditions its
per-source decoders), objective masked-pixel reconstruction plus a
same-place-different-date contrastive term so that the token carries state
rather than texture, output **k = 4–8 tokens of 64 dimensions**, cached once
per tile-date and reused by every anchor whose cone contains the tile.
Snapshot semantics: it sees one date, and the cone codec supplies the time
axis by reading the cached tokens of many dates.

Storage arithmetic, so nobody builds this by accident at the wrong size:
Sentinel-2 over the open ocean is mostly cloud and mostly empty; over land
and coasts (~1.5 × 10⁸ km²) there are ~3.7 × 10⁸ tiles of 640 m, and at ~70
clear dates/yr × 6 tokens × 64 dims × 2 B that is **≈ 20 TB/yr of tokens**
globally — streamable as a tier-P store, but not small. Tiles of 256 px
(2.56 km) cut it 16× to ≈ 1.3 TB/yr and are still 100× finer in area than a
0.25° cell; the tile size is a knob E-078c measures, not a constant. The
catalogue (metadata only) is ~ 10 GB/yr, and the global figure is why the
first build is coastal and regional. Raw imagery is streamed once to
train and once to encode, never kept (the Argo-builder discipline).

What NISAR adds and what it does not: L-band backscatter at 15 m, 12-day
repeat, through cloud and at night, global — the right *snapshot* input for
soil moisture, flooding, ice and vegetation structure, and a fine
complement to Sentinel-1's C band; but data exist only since late 2025, so it
enters as a local-codec input (spatial, per date) and cannot contribute to a
43-year forecast axis. Sentinel-1 (2014→) and Sentinel-2 (2015→) carry the
record; NISAR is folded in by the same local codec with a different `source`
embedding, which is the reason the codec is conditioned on sensor metadata
rather than trained per sensor.

## 5 · Outputs at any granularity — the decoder takes a footprint too

The other half of "varying granularity" is the target side. E-072's recipe
already has a **queryable decoder** (a Gaussian head asked at a location).
It gains the same two footprint fields as the input token: *"the mean
chlorophyll of this 0.25° cell over this pentad"* and *"this 300 m pixel on
this day"* are two different queries of one embedding, and the loss for each
is scored against the observation that actually has that support. This is
what lets one model be trained on tier-G targets today (E-077's `log_chl` is
a valid target with footprint (−2.8→0, 0)) and on tier-T targets later,
without a second head.

A masked-token objective over the §2 schema then covers every tier at once:
withhold a token — a grid cell, a profile, a cached tile token — and predict
its `value[C]` from the rest, at its own footprint. E-076 §7.5's two
objectives (masked-profile reconstruction, next-pentad prediction) are the
tier-P instances of it.

## 6 · What is built first, and the experiment that decides each step

In order, each cheap, each with a falsifier stated now:

1. **E-078a — the footprint fields, on the tensor we have.** Re-read family
   7's coarse groups as *offset tokens* (one token at the coarse cell's
   centre, `(Δx, Δy)` to the anchor, `log2_fp` = +2.9 for NCEP, +2 for
   `rg100`) instead of nearest-cell copies, and add the two footprint fields
   to every sparse token. One cone-codec arm against the E-076a twin on the
   same anchors and seed. Falsifier: no change in held-out per-family loss
   ⇒ the model does not use the fields at this scale, and tier T must be
   argued on its own evidence. Cost: a sampler change and ~$1.
2. **E-078b — `family.json` and the tier dispatch.** Write the registry for
   family 7.1 + family 8 as ONE family (call it family 10 = 7.1 ∪ 8), make
   `cone_sampler` and the app's index reader dispatch on `tier`, and retire
   the per-family npz-key conventions. No model. Falsifier: a test that
   gathers one anchor through the registry and reproduces E-076a's exact
   token bytes.
3. **E-078c — the first local codec, coastal.** Sentinel-2 L2A 10 m tiles
   (2016→) over ~300 coastal 0.25° cells where OC-CCI, OISST and Argo all
   have coverage; train the tile codec, cache its tokens as a tier-P store,
   add them to the cone. Verdict: the masked-token loss on the colour and
   SST targets at those anchors, with and without the tile tokens, three
   seeds (a new tier buys its own replication, `ml/CLAUDE.md` §3b).
   Falsifier: parity ⇒ at pentad cadence the coarse fields already carry
   what 10 m adds, and the fine tier is for fine *targets*, not for the
   forecast. Cost: the only compute-bound rung on the list — tens of
   4090-hours for the tile codec, a few for the cone arm.
4. **E-078d — NISAR and Sentinel-1 through the same codec** with a radar
   `source` embedding, over sea-ice-edge and flood-prone cells, where the
   day-and-night, through-cloud property matters. After 3, not before.

## 7 · What this does NOT change

The pentad bin as the unit of time; the holdout protocol (2009/2017/2023
development years, train ≤ 2020 / test 2021–2024 terminal); the cone
families and reaches of E-071; family 7.1's files, which become tier-G
groups of family 10 unchanged; family 8's store, which becomes its first
tier-P group unchanged. The design adds a registry and a tier, and takes
nothing away.

## 8 · Sources within the repository

- E-076 §2.6 — the footprint token (`ml/plans/E076_family8_nearest_observations.md`)
- E-076b — optimal interpolation from the same five profiles, 21 % over climatology (`ml/plans/E076b_results.md`)
- Embedding-inputs proposal §2 (the 13-rung ladder), §4.2–4.7 (dots, tokens, the local codec, the hybrid) (`ml/figures/geofm_survey/GENERIC_EMBEDDING_INPUTS.md`)
- E-072 §3 (the queryable Gaussian decoder), §6 (read-outs) (`ml/plans/E072_earth_foundation_model.md`)
- E-070 B2 (native-resolution groups) and E-077 (family 7.1) for the tier-G layout
- The family-7 design note's data ladder, "the kilometre rung" (`ml/paper/notes/family7.pdf`)

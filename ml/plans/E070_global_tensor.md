# E-070 · The global tensor (family 7) — what we have, what the globe buys, and the order of work

**Written 2026-09-03**, on Chris's ask of the same morning: *"Think about the
number of data points we have now, expand to the whole globe (if not already
done so), and proceed with global data preparation and training; see if
additional (finer-grained) data would be useful in the next data family (eg
radar … do we have surface colour somewhere?); is now the time (because we now
have velocities and cones) to move to also support daily data at least in the
embedding's input?"*

Status at writing: **Phase A is running** — the four-lane global GLORYS12
pull, [`glorys-pull-global.yml`](https://github.com/blauewelt/earth/actions/workflows/glorys-pull-global.yml),
dispatched 10:59Z 09-03 after a measured one-month smoke test (§4). Nothing
else in this document exists yet.

Companion documents:
[the data ladder](https://blauewelt.github.io/earth/docs.html?f=ml/plans/DATA_LADDER.md)
(every candidate channel, priced, 08-30),
[the E-033 scale programme](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E033_scale_program.md)
(§4 "global adds independent samples", §5 the storage moves, Phase 4 = this),
[the E-069 cone codec](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E069_cone_codec.md)
(the codec family 7 is built for),
[the protocol reset](https://blauewelt.github.io/earth/docs.html?f=ml/plans/PROTOCOL_RESET.md)
(what may count as a result).

---

## 0 · The one-paragraph answer

The North Atlantic tensor holds ~2.5 billion observed values over 84,405
ocean cells and 3,142 pentads, and every wall the programme has measured is a
wall of **distinct temporal samples**, not of pixels: the stage-2 heads reach
their best held-out loss at ~step 2,000 whatever their size, the capacity
ladder is flat across 27× in parameters, and a 200-mode linear model beats
every head after two pentads. The globe does not add a single temporal sample
to the RAPID question — the labels are Atlantic and the end-bins are the same
2,417. What it adds is **8.1× the anchors, from dynamical regimes the window
has never shown the codec** (the ACC, the Kuroshio, the equatorial waveguide,
the Arctic seasonal ice edge), which is the one thing a *representation* —
as opposed to a transport head — is starved of. So family 7 is a stage-1
investment: it is how the "obs-only ocean-interior embedding" that the survey
listed as not-done-by-anyone gets its data, and its first gate is the one
E-033 wrote in advance — *a globally trained codec must match or beat the
NA codec on the Atlantic tasks* (it matched exactly at 1°, run #8). It is
cheap in compute (a cone codec samples anchors; 20k steps cost the same on
any tensor) and expensive only in data engineering, all of which is the
credential-safe runner machinery E-034 already built. The rest of the ask —
finer pixels, radar, colour, daily — is answered in §6–§8: statics and radar
*altimetry* yes, colour as a target not an input, finer pixels no, daily as a
**multi-rate inner cone** after H1, not as a daily tensor.

---

## 1 · What we have now — the inventory, honestly counted

The tensor every current number was measured on is family 4, recipe r2
(`family4_na025_pentad_r2`, sha `37e14638…`): `[3142, 281, 481, 40]`
float16, 0.25° point-aligned, 100°W–20°E × 0–70°N, 5-day bins from
1982-01-01, 33.3 GB. Recipe r3 (+`cur_u`, `cur_v`, 42 channels, 35.7 GB) is
the E-069 blocker and has never been built.

| what | count | note |
|---|---|---|
| grid cells / ocean cells | 135,161 / **84,405** | 62.4 % ocean |
| pentad bins / bins with GLORYS | 3,142 / 2,339 | 1982–92 carries wind + SST only |
| train end-bins (frozen protocol, ≤ 2020, dev blocks out) | **2,417** | the number that never changes |
| train windows in the stage-2 pool | 209.5 M | = 2,417 × 86,698 |
| observed values, base 3 ch (1993–2024) | ≈ 0.59 B | 3 × 2,339 × 84,405 |
| observed values, wind 4 ch (full axis) | ≈ 1.06 B | 4 × 3,142 × 84,405 |
| observed values, SST (r2, full axis) | ≈ 0.27 B | live in ~100 % of bins |
| observed values, RG 32 ch | ≈ 0.6 B | 252 live bins × 32 × ~85 % of cells; **1°-smooth, carries ~0.28 GB of information in 27 GB** |
| **total observed values** | **≈ 2.5 B** | the codec-side Chinchilla anchor is values/20 ≈ 125 M params, a *ceiling* (cells are correlated over hundreds of km) |
| RAPID / Florida-cable pentad labels | 1,459 / 2,553 | never inputs |

The builder prints the exact inventory at build time; the figures above are
arithmetic from the grid and are meant to be checked against it.

**The three walls, and which one the globe touches.**

1. *The head's wall is temporal.* 7.6 M, 40.4 M and 206.7 M heads reach the
   same mean corridor ACC (0.103 / 0.104 / 0.105, E-062) and the LIM beats
   all three from 15 days out (E-066). More pixels of the same 43 years do
   not move this (data ladder §0), and neither does the globe: the Atlantic
   head still sees 2,417 end-bins.
2. *The codec's wall is anchors × regimes.* The cone codec reads 748 tokens
   around one anchor; its training set is the set of anchors, and 84,405
   cells × 2,417 bins of one basin is one climate's worth of eddy statistics.
   The globe multiplies anchors by 8.1 and, more importantly, adds regimes
   that are not more of the same. This is E-033 §4's argument and it is the
   only argument for going global; it is a good one.
3. *The probe's wall is labels.* 240 monthly RAPID labels; nothing here
   changes it, and family 7 must not pretend to.

---

## 2 · Decisions taken (all reversible; say so in a reply and they flip)

| # | decision | why | what it costs |
|---|---|---|---|
| D1 | **Source = GLORYS12 1/12° daily, binned to 0.25° at fetch time** (not the 0.25° GREP ensemble member) | family 4's base channels came from exactly this product through exactly `aggregate_cadence.bin_plan/bin_slice`; the global tensor's NA sub-block is therefore computed by the code that made family 4, and a global-vs-NA codec comparison is a comparison of codecs, not of model streams | 840 GB on the wire instead of ~51 GB; ~1 day of runner time in four lanes; 99 GB on the Hub instead of 840 |
| D2 | **Grid = point-aligned 0.25°, lat −80…90 (681), lon −180…179.75 (1440)**; NA is the exact sub-block at global index (320, 320) | GLORYS12's own extent; verified on the real 1993-01 chunk | 980,640 cells, ~686 k ocean (est.) |
| D3 | **Channel layout = r3 (42), same order; the 32 `rg_*` channels live in a 1° SIDECAR, live bins only** | the cone reads `rg_*` at the anchor column only (lags 1–6, `ml/cone.py:256-274`), so a 1° cell lookup is *exactly* what the model consumes; upsampled into the tensor they would be 197 GB of the 259 | dense tensor 61.6 GB + 0.84 GB sidecar instead of 259 GB; fits the 128 GB-RAM boxes; ~40 release parts instead of 173. The lag-0 patch for `rg_*` is served as the same 1° cell nine times (token shape unchanged, physically honest — RG has no sub-degree structure) |
| D4 | **Dateline wraps** (`wrap_x`) in `cone_sampler`; the window-is-a-basin rule that forbids wrapping is correct for NA and wrong for the globe | a Kamchatka anchor reads the Aleutians | ~10 lines; a test |
| D5 | **Stage 2 stays Atlantic.** The codec trains globally; `Z` is cropped to the NA sub-block before any head, LIM or roll; the corridor percentile is taken over the NA rectangle; the #217 gate gets a new reference or `--no-gate` | the labels are Atlantic; embedding the globe would cost 85–170 h of 4090 for nothing the probe can read | the E-033 Phase 4 gate becomes runnable at all |
| D6 | **Cadence = pentad.** Daily is §8 | the daily tensor is 165.6 GB for NA alone; 1.3 TB global | — |
| D7 | **Protocol = the frozen one** (PROTOCOL_RESET §3.1): train ≤ 2020, test 2021–2024 as one terminal block, opened once on Chris's word; the development blocks 2008–09 and 2016–17 held out in the same codec; `holdout_years = 2008,2009,2016,2017,2021,2022,2023,2024`, `--holdout-scope window`. The interspersed 2009/2017/2023 split is development only (#537) | nothing in the holdout machinery is keyed to the grid (audit §5) | — |
| D8 | **Transport = the Hub + the bucket, no streaming loader yet.** The 62 GB tensor is memmapped from local NVMe on a 500 GB-disk box; E-033's object-store block-cache reader (Phase 3) stays deferred because D3 made it unnecessary at this size | a 500 GB-disk box class for builds and training (offers exist at the 4090 price) | the loader is owed the day the *daily* global family is attempted |

---

## 3 · What the globe buys, stated as falsifiable claims

**G1 — the generic-embedding hypothesis at 0.25°.** A cone codec trained on
family 7 (same recipe as E-069's `f4r3-cone-7M`, same seeds, same steps)
embeds the NA sub-block at least as well as the NA-trained codec: H1's
velocity probe (ridge R² from frozen `z` to `cur_u`/`cur_v`, inputs hidden)
agrees within the three-seed interval, and the 7.6 M head through the #516
battery agrees within the five-seed interval. *Falsified if the global codec
is worse by more than the interval on either* — which would say the basin's
statistics are specific and global pretraining costs the Atlantic, the result
run #8 ruled out at 1°.

**G2 — regimes are samples.** At matched steps, the global codec's held-out
one-step NLL *on the NA sub-block* is lower than the NA codec's, or the
velocity probe is higher, by more than the seed interval. *Falsified if they
agree* — then the globe was 8× more of the same and the honest conclusion is
that the codec was never anchor-starved, a real finding that closes the axis.

**G3 — the velocity content transfers.** H1 holds *outside* the window: the
frozen global `z` predicts `cur_u`/`cur_v` in the Kuroshio and ACC boxes with
R² in the same 0.3–0.6 band predicted for the Gulf Stream. *Falsified if R²
collapses in a basin the codec was trained on* — the embedding would be
learning the Atlantic's velocity field, not velocity.

Pre-registered: G1 holds, G2 is the open question (I would bet on a small
positive NLL effect and a null probe effect), G3 holds. Each gets §3b's
replication only where variance lives — three seeds on the codec, the pair on
the head.

---

## 4 · Phase A — the pull (COMPLETE)

**2026-09-04.** Phase A is done: all **384/384** monthly chunks stood under
`daily025_global/` on the Hub on 2026-09-03, four lanes at 96/96 each, and the
gate below is met as written.

`ml/fetch_glorys_daily.py --window global --bin-deg 0.25` fetches each month
day by day (a whole month in one `cm.subset` peaked at 5.95 GB and was
OOM-killed on the 7 GB runner), bins every daily slice through
`aggregate_cadence.bin_plan/bin_slice`, writes one float32 NetCDF per month
and stores it under `daily025_global/` on
[`chfrank/earth-tensors`](https://huggingface.co/datasets/chfrank/earth-tensors),
restore-verified, deleted locally. Measured on the real 1993-01 chunk from
the sandbox: **2,188 MB on the wire, 257 MB stored, 411 s fetch + bin,
peak RSS 1.9 GB.** Four lanes (1993–2000 · 2001–2008 · 2009–2016 · 2017–2024),
each its own concurrency group, 6-hourly cron plus the manual dispatch that
started them at 10:59Z; ~96 chunks × ~8 min per lane ≈ 13 h of work, i.e.
three firings — **about a day**. The NA chunks in the dataset root are
untouched. Hub quota: PRO to 2026-10-01 (10 TB); the dataset was 155.7 GB
before this and lands at ~255 GB; the dataset is public, so after PRO lapses
it is on the Hub's best-effort public tier — check before relying on it.

*Gate:* 384 chunks under `daily025_global/`, each lane's tally at 96/96.
*Failure to watch:* a lane whose tally does not advance across two firings
(a month that fails five times stops the lane by design).

## 5 · Phases B–F, with what stops each

**2026-09-04 — Phases B, C and D below are SUPERSEDED.** They are built as one
resumable job rather than as three, by
[the family-7 build spec](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E070_family7_build.md)
(recipe `f7l0`): the other channels, the pentad aggregation and the publish are
stages of `ml/build_family7.py` on a Vast box, which also takes the grid to
both poles and adds the shared land/ocean channels E-071 §6 asks for. Read the
three phases below for what each step must produce and what stops it — that
part still holds; read the spec for how it is actually run. Phases E and F are
unchanged.

**Phase B — the other channels, global (runner work, ~$0).**
`rg_*` and NCEP are already global files (audit §1): only the destination
grid changes, and the `lon360` shift becomes the plain `lons<0 → +360` form
the wind path already uses. OISST needs `fetch_sst_na.target_grid` to accept
the global axes and the output chunked by year (a 30.8 GB single member
would break the restore-verify; 43 × 0.72 GB does not). Same ~20 GB download
as today, same `sst-na-bake.yml` shape. *Stop if:* the runner's 87 GB cannot
hold a year plus its verify copy — then bake on a box from the Hub instead.

**Phase C — pentad aggregation (runner, split by year).** `aggregate_cadence
--hf-prefix daily025_global/` with *no* `--bin-deg` (already on the grid);
~14 h global against the 350-min cap, so four year-lanes like Phase A.
Output `pentad025_global/`. *Gate:* `has_data` covers 2,339 bins; the NA
sub-block of `pentad_mean_uo` equals family 4's `pentad_mean_uo` to float32
(a bit-identity test — the binning is the same code, but the order of the
time mean over binned days vs binning of time means is not guaranteed
identical; measure it, and if it is not bit-identical record the max |Δ|).

**Phase D — the build (a 500 GB-disk box, ~6 h, ~$3).** `build_family7.py`
as the `build_family5.py`-style wrapper over `build_family4.py`: global grid
reference (`--grid-ref`, *keep the refusal*), `RECIPE_REV = "f7r3"`, sidecar
on, the rg block written to its own 1° sidecar with `lat_rg/lon_rg/
rg_bin_index`, `window: "global025"`. Publish as 1.5 GiB parts to the Hub
and to `gs://earth-tpu-staging/tensors/`; pin the sha in the workflow and the
launcher; **build once** (E-069 handover's trap: two builds are two
experiments). *Stop if:* the NA sub-block's per-channel z-scores differ from
family 4's by more than the fp16 quantum after accounting for the global
normalisation — then something upstream is not the same bytes.

**Phase E — the codec and the three gates (~$25 all in).**
Three seeds of `f4r3-cone-7M` on family 7 (GPU via workflow or TPU via
`tpu_train_cone.sh` — a TPU number is its own §3b tier and buys a torch
twin), 20k steps, batch 256, `holdout_years = 2008,2009,2016,2017,2021,2022,2023,2024` (the frozen protocol: train ≤ 2020; recipe `f4r3-cone-7M-terminal`), the pool
certificate at 0 violations, `velocity_probe` on. Then: embed the **NA crop
only** (10–20 h of 4090, the existing `embed_cone` estimate), the 7.6 M head
under E-064b's configuration, the #516 battery, and G1–G3 read against
E-069's own NA-codec seeds — *which is why E-069 Phase C (three NA seeds on
r3) must land first; family 7 is not a substitute for it and must not delay
it.* *Stop if:* G1 fails — record it, keep the NA codec, and the global
family becomes a pretraining corpus for E-061-style fine-tuning rather than a
replacement.

**Phase F — only if E passes: the capacity rung.** The anchor moves 8×
(~20 B values); one rung (7 M → ~40 M cone codec, the family-3 anchor size)
at matched steps, three seeds. *Stop if* it agrees with 7 M within the
interval — the E-062 lesson (capacity is not the axis) would then hold for
the codec too, and that is the finding.

Order against the rest of the board: Phase A–C need no decision and no
money. Phase D needs the r3 layout to have been built once on NA (E-069
Phase A) so the two tensors share a recipe; Phase E needs E-069's NA seeds.
Nothing here pre-empts the terminal-holdout codec (E-068) or the block rolls
(E-067), which remain the ranked next steps of the 09-03 handover.

---

## 6 · Finer pixels, radar, colour — the next family's channels

The data ladder priced every candidate for the NA window on 08-30 and its
ranking stands; what the global move changes is the *cost per channel* (one
dense global pentad channel is 6.16 GB fp16, 7.3× the NA figure) and the
*purpose* (a representation of the ocean, not only a transport predictor).
Read in that light:

**Finer pixels — no, and the cone says why.** At pentad cadence a cell of
0.25° is the smallest displacement the codec can resolve per lag (0.06 m/s).
At 1/12° the same physics needs ~9× the dots for the same kilometre reach,
and the tensor is 305 GB for NA alone (ladder §4). Resolution without cadence
buys nine correlated views of the same 2,417 bins. The resolution experiment
that is worth running is the ladder's **23–30°N strip at 1/12°** (18 GB): it
asks whether the RAPID *section* cares, which is the only place the answer
could change a number we report.

**Radar — yes, the gridded kind.** Three radar instruments produce products
that fit a 0.25° pentad grid; the fourth does not:

| radar product | what it is | fits? | verdict |
|---|---|---|---|
| **Altimetry — DUACS L4** (`SEALEVEL_GLO_PHY_L4_MY_008_047`, 0.125°, daily, 1993→) | radar altimeters: `sla`, `adt`, `ugos`, `vgos` + errors | yes, at 0.25°: ~4 ch × 6.2 GB global, 3.8 GB NA | **first radar channel to add.** The only *observed*, model-independent surface circulation; every current base channel is GLORYS output. Two baselines (`sla` vs `zos`), so two channels, not one |
| **Scatterometry** — ASCAT/QuikSCAT L4 winds (`WIND_GLO_PHY_L4_MY_012_006`, 0.25°, 1999→; or ERA5 which assimilates them) | radar-derived surface wind stress | yes | second; but ERA5 stress (ladder C1) covers the whole axis from 1940 and needs only a free CDS account — **that account is the one action only Chris can take** |
| **Precipitation radar** (IMERG, 0.1°, 2000→) | E−P forcing | yes at 0.25° | only via ERA5's `mtpr`/`mer` at this cadence; IMERG's value is sub-daily |
| **SAR** (Sentinel-1 / NISAR backscatter, 10–30 m swaths) | imagery | **no** — swaths, no gridded long record, no per-pixel state; its L4 derivatives (ice, waves, wind) already exist as the products above | not a tensor channel; it stays a globe layer |

**Surface colour — we have it on the globe, not in the tensor, and that is
the right place for now.** The app already renders PACE OCI chlorophyll-a
and MODIS true colour (GIBS); no ML family has a colour channel. The ladder
skipped ocean colour for AMOC ("no mechanistic path at these timescales"),
and that holds. For the *Earth-2* ambition it is different: OC-CCI v6
(4 km, 1997→, the merged SeaWiFS/MODIS/MERIS/VIIRS/OLCI record) or the
CMEMS GlobColour L4 (`OCEANCOLOUR_GLO_BGC_L4_MY_009_104`, 4 km → 0.25°) is
the only global observation of the biosphere, and the survey's biosphere
chapter found chlorophyll predictable for about one season with subsurface
memory the key. So colour enters family 8 as a **decoder target and probe**
(does the physical embedding predict next-season chl-a?) before it enters as
an input; as an input it is clear-sky-and-daylight only, which the
missing-token design handles but which halves its live bins at high
latitude. Cost at 0.25° pentad: 6.2 GB global, one channel.

**The order for family 8, then:** the ten statics (bathymetry, slope, mask,
f, β, distance to coast/1000 m isobath, MDT — 0.27 MB each *per channel
globally*, the highest information per byte on the list; a land/ocean mask
finally distinct from "not observed"); DUACS; ERA5 once the CDS account
exists (retiring NCEP's 2.5°); OSTIA SST + sea-ice fraction (the Labrador /
Irminger / Arctic edge that a global tensor makes matter); then GREP 3-D at
8 levels only if the depth question is asked globally (20 GB NA → 150 GB
global — that one wants the streaming loader first). Colour as a target.

---

## 7 · Daily in the embedding's input — the cone makes this a design, not a rebuild

Yes, and the cone is what makes it affordable — but not as a daily tensor.

The inner cone is already **per-family**: family A reads lags 0–1 (its 10-day
memory), B and C read 0–6, `rg_*` reads a column. The same machinery admits
per-family *cadence*. The channels that are truly daily-informative are the
ones observed daily on the grid: SST (OISST) and wind stress (NCEP/ERA5) —
families C and A. GLORYS is a model's daily output, RG is monthly. So the
proposal is a **multi-rate inner cone** — this is the survey's "multi-rate
tokens per sphere" design implication, made concrete:

| family | inner cadence | lags | reach at lag l | dots / channel (est.) |
|---|---|---|---|---|
| A (wind) | **daily** | 0–10 (its 10-day memory) | 500 km floor | ~70 (vs 8) |
| C (SST, MLD) | **daily** | 0–30 (the same 30-day window) | 25.9 × (1+l) km, floor 100, 500 at l ≤ 5 | ~300 (vs 81) |
| B (currents, SSH) | pentad | 0–6 | 129.6 × (1+l) km | 80 (unchanged) |
| rg | pentad | column | — | 6 |

Dot budget ≈ 4×70 + 4×80 + 32×6 + 2×300 ≈ **1,400 vs 706** — twice the
tokens, not thirty times, because only the daily-observed families pay. The
data is a **daily sidecar for six channels** beside the pentad tensor: NA
SST daily already exists (`sst_na025/`, 4.3 GB int16) and NCEP is daily at
source (4 × 4.25 GB) — ~21 GB beside the 35.7 GB r3 tensor, on the existing
fleet. Globally the same six channels are ~185 GB, which is the daily
family's real cost and the reason it is sequenced after, not with, family 7.

**When:** after E-069's H1 has been read on the NA r3 codec — H1 is the
question "does velocity in the embedding buy anything", and the multi-rate
cone is the same question at a finer rate; asking both at once confounds
them. It is E-071 by number and needs its own plan: the aggregation identity
for the daily σ channels (`tau_*_std` is a within-pentad σ, so at daily rate
it is the centred 5-day σ family 5 already defines), the window-scope
certificate at mixed rates, and whether the decoder's future targets stay
pentad. What it does NOT need is the 165 GB daily tensor or the streaming
loader: the sidecar is the point.

---

## 8 · What is deliberately not done here

- **E-069 Phase A (the NA r3 build) is not started by this document.** It is
  the E-069 session's lane and a second builder would make two tensors with
  two shas; but it is the prerequisite for Phase D and E above and remains,
  as the overview says, the single blocker on the whole experiment.
- **The streaming loader** (E-033 Phase 3) is deferred, not cancelled; the
  rg sidecar is what makes deferral honest at 62 GB.
- **No global stage 2.** No labels, no purpose, 85–170 h of embedding.
- **No family-7 result is comparable with a family-4 number** except through
  the NA-crop protocol of D5; the house rule on tensor generations applies.

## 9 · Costs and the runway

Phases A–C: $0 (GitHub-hosted runners; Hub storage under PRO to 10-01).
Phase D: one 500 GB-disk box for ~6 h ≈ $3 + parked storage. Phase E: three
codec seeds ($3–7 each on TPU spot, $2–3 on a 4090), the NA-crop embedding
(~$3), one head pair and the battery (~$5) — **≈ $25**. Credit was $35.15 at
07:02Z 09-03 with auto top-up on; per `ml/CLAUDE.md` §0e the arithmetic is
reported, the fleet is never parked over it.

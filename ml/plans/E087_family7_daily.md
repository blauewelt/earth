# E-087 · Family 7.2 at daily resolution — what it takes, proved on one month

*Written 2026-10-06, after a real one-month probe (January 2015) from the
sandbox and BEFORE any build. Chris asked: "Please build the per-year monthly
files. Could we have daily files as well?" — the monthly files are another
agent's work; this plan answers the second half. Nothing has been dispatched,
rented or uploaded.*

## 0 · In plain English

- **Family 7.2** (`family7_global025_pentad_l2` on the Hugging Face dataset
  `chfrank/earth-tensors`) is the global input tensor of this programme: 56
  channels on a 0.25° and a 1° grid, pole to pole, 1982–2024, every value a
  **five-day mean** (a "pentad": the fixed five-day bins counted from
  1982-01-01). Its channels come from four public sources that are all
  **daily** underneath, plus one monthly product:
  - **GLORYS12** — Mercator Ocean's ocean reanalysis (a numerical ocean
    model pulled towards the observations): surface currents, sea-surface
    height and mixed-layer depth, from 1993.
  - **OISST v2.1** — NOAA's daily optimum-interpolation analysis of satellite
    and in-situ sea-surface temperature and sea-ice concentration, from 1982.
  - **NCEP/NCAR Reanalysis 1** — the US weather reanalysis: wind stress,
    air temperature, wind, pressure, rain, snow, soil, heat fluxes and skin
    temperature, four analyses a day, from 1948.
  - **OC-CCI v6.0** — ESA's merged satellite ocean-colour record
    (chlorophyll-a at 4 km), from 1997-09-04.
  - **Roemmich–Gilson Argo** — a MONTHLY gridded product of ocean temperature
    and salinity at depth (the `rg100` group). It has no daily form.
- **The daily version** is the same channels, the same grids, the same
  derivation, minus the last averaging step: one value per DAY instead of the
  mean of the days in a five-day bin.
- **The layout** is the family-1 "sharded tier-G" layout
  (`ml/family1/sharded.py`): each five-day bin is one compressed file holding
  its five daily frames as independently compressed 256 × 256 tiles (64 × 64
  at 1°), so a reader fetches a region and a day without the rest. It is the
  layout the Data tab already reads (ERA5 since yesterday, ocean colour at 4 km
  before that).
- **The promise the daily store makes**: average its five frames the way
  family 7 averages and you get the published pentad value back, to float16
  rounding, everywhere the pentad has a value — and NaN exactly where it is
  NaN. The probe measured that promise on January 2015 for all four sources
  and **it holds in every cell of every channel** (§4).
- **Size and cost**: about **117 GB** compressed (259 GB as raw float16), free
  hosted-runner lanes plus one cheap assembly box, **about $1** and about a
  day of wall time — a third of yesterday's ERA5 build (349 GB, $1.76).
- **No credential is needed anywhere.** The global daily GLORYS fields are
  already parked on the Hub, complete, so the Copernicus Marine credentials
  are not touched.

Links:

- [the probe and derivation module](https://github.com/blauewelt/earth/blob/main/ml/family7_daily.py)
- [its tests](https://github.com/blauewelt/earth/blob/main/tests/test_family7_daily.py)
- [the January-2015 probe report (JSON)](https://github.com/blauewelt/earth/blob/main/ml/plans/E087_probe_2015-01.json)
- [the family-7 build spec, E-070](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E070_family7_build.md)
- [the ERA5 precedent, E-085](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E085_family12_atmosphere.md)

## 1 · Decisions — recommended picks, each reversible

| # | decision | recommended pick | why |
|---|---|---|---|
| D1 | stores | **one store per SOURCE**: `glorys025d` (5 ch), `oisst025d` (2 ch), `ncep100d` (15 ch), `occci025d` (2 ch) | a missing or slow source never blocks the others (OC-CCI is ~1 h a year of CEDA transfer, NCEP is minutes); the sharded layout gives one grid and one channel list per store; the four records start on four different days |
| D2 | channels | **identical names, units and derivation to family 7.2**, except "the day's value" where 7.2 says "mean of ≥ 3 days" | every derivation in `ml/family7_daily.py` calls the family-7 builder's own helpers in the builder's order of operations; nothing is re-derived |
| D3 | `tau_x_std`, `tau_y_std` | **population σ of the 6-hourly NCEP samples of the five days CENTRED on the day** (≥ 3 of those days present) | a one-day σ of a daily value is identically zero; the centred window keeps family 7's meaning and makes **frame 2 of every bin exactly that bin's pentad σ** (measured, §4). Family 5 (the North Atlantic daily tensor) chose the same centred window. The alternative — the σ of a day's four samples — is a different quantity (open question Q3) |
| D4 | units | **physical units, float16** (family 1's convention; the pentad norm is copied into `store.json` for whoever wants z-scores) | measured float16 error is ≤ 2.5 % of the day-to-day standard deviation for 22 of 24 channels and 7–9 % for `sst` and `sp` (§5.3) — one to two orders below each source's own analysis error. Z-scoring with the pentad's published norm (fixed constants, so no extra pass) would cut `sst`'s and `sp`'s error 2.7× and is the measured alternative (Q2) |
| D5 | NaN | the source's own: GLORYS NaN on land and south of 80° S; `sst` NaN where OISST does not observe; `sea_ice` NaN below 15 % ice (OISST's mask, as in 7.2 — open water is NaN, not 0); `soilw`/`tsoil` NaN over sea (gaussian land mask before regridding); `log_chl`/`chl_cov` NaN where no 4 km cell of the 0.25° block was clear that day | identical to family 7.2's §5 semantics; the ≥ 3-day rule does not apply to a single day |
| D6 | record | **family 7.2's**: to 2024-12-31; from 1993-01-01 (GLORYS), 1982-01-01 (OISST, NCEP), 1997-09-04 (OC-CCI). Frames outside are `before_record`/`after_record`; a day missing upstream is `absent_upstream` (the layout's own reasons) | the consistency guarantee is with 7.2; the GLORYS chunks on the Hub stop at 2024-12. Extending is Q4 |
| D7 | grid, tiles | 0.25° 721 × 1440 and 1° 181 × 360, south-first, point-aligned (7.2's grids); **256-pixel tiles at 0.25°, 64 at 1°** (both 64° on a side; 18 tiles a frame) | E-085's measurement: 64 at 1° is more local and smaller |
| D8 | `frames_per_bin`, `frame_seconds` | 5, 86,400 | frame f of bin b is day 5b + f after 1982-01-01 |
| D9 | inputs | GLORYS: the parked `daily025_global/` chunks on the Hub. OISST, NCEP: the Hub's PSL mirror (`mirrors/psl/`), PSL as fallback. OC-CCI: CEDA 1997–2022 + PML's per-day subset 2023–2024 — **the same files and the same block reduction 7.2's `oc025` used** | bit-faithful inputs are what make the falsifier exact; all keyless |
| D10 | what is not carried | `rg100` (monthly), the statics `sphere`/`elev` and the truth series stay in the pentad npz | a static has no daily axis; daily truth is Q5 |
| D11 | family | proposed code **`72d`** → `tensors/family7_2d/<store>/`, version "7.2d", registry `tensors/family7_2d/family72d.json` | follows family 1.2's pattern (`ml/family1/adapters/__init__.py` FAMILIES); the name is Chris's (Q1) |
| D12 | tile byte order | C order, as `sharded.py` writes (E-085 D2) | a planar/shuffled option would save ~20 % but is a format fork across every reader |

## 2 · The measured inventory — where the daily inputs are

Listed 2026-10-06 from the Hub's tree API (anonymous) and by HEAD/GET against
the producers, from this sandbox.

| source | daily inputs today | complete? | bytes | grid | credentials |
|---|---|---|---|---|---|
| **GLORYS12** (`uo`, `vo`, `mlotst`, `zos`) | **Hub `daily025_global/glorys025_global_YYYYMM.nc`** — the global 1/12° daily product binned to 0.25° at fetch (E-070) | **yes: 384 of 384 months, 1993-01 → 2024-12**, none missing; 233–262 MB each | **96,800,818,463 B** | 681 × 1440, lat −80…90, lon −180…179.75, one value per day per variable (opened: 2015-01 has 31 days, 2015-02 28; float32, zlib; `source_dataset` `cmems_mod_glo_phy_my_0.083deg_P1D-m`) | **none** (public Hub read). Re-fetching from Copernicus Marine is NOT needed — it would be 840 GB at 1/12° and need the Actions secrets |
| **OISST v2.1** (`sst`, `icec`) | **Hub `mirrors/psl/Datasets/noaa.oisst.v2.highres/`** (and PSL itself, keyless; `sst.day.mean.2015.nc` 475.7 MB in 11 s, `icec` 73.1 MB in 3 s here) | **yes: 43 + 43 files, 1982–2024** | 23,479,739,136 B | 720 × 1440 cell-centred, 365/366 days a file | none |
| **NCEP/NCAR R1** (13 surface variables + land mask) | **Hub `mirrors/psl/Datasets/ncep.reanalysis/surface_gauss/`** (and PSL, keyless; 13 files of 2015 in 16 s here) | **yes: 13 × 43 files 1982–2024 + `land.sfc.gauss.nc`** = 560 | 21,387,740,208 B | T62 gaussian 94 × 192, **4 samples a day** (2015: 1,460 steps) | none |
| **OC-CCI v6.0** (`chlor_a` 4 km) | **NOT parked as dailies at 0.25°.** The Hub has (a) `partials/f7l1/occci/` — 28 per-YEAR files of five-day ACCUMULATORS (sums over days per pentad bin, 20.9 GB), not daily; (b) the family-1.gf store `tensors/family1_gf/oc4k/` — daily 4 km `log_chl`, `kd_490`, `total_nobs`, float16, **1997-09-04 → 2022-12-31 only**, 3,702 shard files, 175.96 GB. The dailies must be re-read from the producer: CEDA `dap.ceda.ac.uk/…/chlor_a/daily/v6.0/<Y>/` (1997–2022, ~80 MB a day; 30 files of Jan 2015 = 2.41 GB in 34 s here, 4 connections) and PML's NetcdfSubset of `CCI_ALL-v6.0-DAILY` one day at a time for 2023–2024 (E-077) | the producers are complete; nothing daily at 0.25° exists | ~790 GB of transfer (E-077) | 4320 × 8640 at 1/24°, north-first | none |
| Roemmich–Gilson Argo | monthly only | — | — | — | — (excluded, D10) |

## 3 · The design

### 3.1 The four stores

| store | grid · tile | channels (unit) | the day's value | daily meaningful? |
|---|---|---|---|---|
| `glorys025d` | 0.25° · 256 | `cur_speed` (m/s), `log_mld` (log₁₀ m), `ssh` (m), `cur_u`, `cur_v` (m/s) | the chunk's value for that day; `log_mld` = log₁₀ of that day's depth where > 0; `cur_speed` = hypot of that day's u, v | yes — GLORYS12 is a daily-mean product |
| `oisst025d` | 0.25° · 256 | `sst` (°C), `sea_ice` (fraction, NaN below 0.15) | bilinear of that day's OISST field to the point grid (`f3.interp2_nan`, wrap 360), exactly the field `stage_sst` accumulates; the `icec` file's 0…1 range trusted over its "percent" string | yes (an analysis: day-to-day changes are partly analysis increments) |
| `ncep100d` | 1° · 64 | the 15 g100 channels: `tau_x`, `tau_y` (N/m², sign-flipped), `tau_x_std`, `tau_y_std` (D3), `t2m` (°C), `u10`, `v10` (m/s), `sp` (hPa), `log_prate` (log1p mm/day), `log_swe` (log1p mm w.e.), `soilw` (fraction), `tsoil` (°C), `lhtfl`, `shtfl` (W/m²), `skt` (°C) | the NaN-aware mean of the day's four 6-hourly samples ON THE NATIVE GRID, then 7.2's bilinear and transform | yes — the reanalysis is 6-hourly |
| `occci025d` | 0.25° · 256 | `log_chl` (log₁₀ mg/m³), `chl_cov` (fraction of the block's 4 km cells clear) | the day's block mean of log₁₀(chl) over the clear 4 km cells (`oc_block_stats`), the very term 7.2 adds to its accumulator; `chl_cov` = clear cells / block cells | yes but sparse: **19.6 % of all cells, 29 % of ocean cells, valid on a January 2015 day** (measured; the 4 km field itself is ~11 % valid) |

Every store: float16, physical units (D4), `frames_per_bin` 5,
`frame_seconds` 86,400, NaN = not observed, bounds are sanity envelopes
(outside → NaN and counted; the probe counted **zero** out-of-bounds values in
all 24 channels).

### 3.2 How each pentad value is rebuilt from five frames (the falsifier's rule)

| channels | family 7.2's pentad | rebuilt from the daily frames |
|---|---|---|
| linear (`ssh`, `cur_u`, `cur_v`, `sst`, `sea_ice`, `tau_x`, `tau_y`, `t2m`, `u10`, `v10`, `sp`, `soilw`, `tsoil`, `lhtfl`, `shtfl`, `skt`) | mean of the finite days, NaN below 3 | the same |
| `cur_speed` | hypot of the MEAN u and v (not the mean speed — the handover's wording "mean of the daily speeds, hypot" is ambiguous; the code is hypot-of-means, and the probe confirms it) | `hypot(mean cur_u, mean cur_v)` |
| `log_mld` | log₁₀ of the MEAN depth | `log10(mean(10 ** log_mld))` |
| `log_prate`, `log_swe` | log1p of the mean rate / water equivalent | `log1p(mean(expm1(x)))` |
| `tau_x_std`, `tau_y_std` | σ of the bin's 20 six-hourly samples | **frame 2** (its centred window IS the bin) |
| `log_chl` | mean of the daily block means over the days with ≥ 1 clear cell | the same, ≥ 1 day |
| `chl_cov` | clear cell-days / (cells × 5) | sum of the finite days' `chl_cov` / 5 |

## 4 · The falsifier, and what it returned on January 2015

**The guarantee.** For every bin, channel and cell, rebuild the pentad value
from the five stored daily frames by §3.2's rule and compare it with the
published family-7.2 value, un-z-scored with the published norm. The store is
WRONG if (i) the NaN pattern differs anywhere, or (ii) any difference exceeds
the rounding both values have already been through:

- half a float16 step of the published z-score × the channel's sd;
- **for g025 only**, half a float16 step of the physical value as well —
  found by this probe: `build_family7` fills `g025` as a float16 memmap in
  physical units and z-scores it **in place** (`RAW_F32` lists only g100,
  rg100 and oc025), so every published g025 value has been rounded twice;
  without this term 36 % of `sst` cells "failed" by up to 2.4 half-steps;
- 8 float32 half-ulps of the largest summand in the source's own units (K,
  Pa — `interp2_nan` returns float32 and both paths do float32 arithmetic);
- the daily store's own float16 rounding (the largest half-step of the five
  frames; × 1.5 for `cur_speed`).

**The probe.** Bins 2411–2416 (2015-01-03 → 2015-02-01, the six bins whose
first day is in January, 30 frames), real inputs, the real sharded writer
(`ShardWriter.write_bin`), read back through `ShardedGroup.read_frame`, the
published pentad slabs read from the Hub by one HTTP range request per group
(offsets and norm from `data/family7_index.json`).

| store | cells compared per channel (6 bins) | NaN pattern | rebuilt from the stored frames: within tolerance | max \|daily-mean − pentad\| (physical) |
|---|---|---|---|---|
| `glorys025d` | 4.16 M | identical, all 5 channels | **100 %** | cur_speed 0.0012 m/s · log_mld 0.0020 · ssh 0.0015 m · cur_u 0.0013 · cur_v 0.0011 m/s |
| `oisst025d` | 4.22 M (sst) · 0.90 M (sea_ice) | identical | **100 %** | sst 0.018 °C · sea_ice 0.0005 |
| `ncep100d` | 0.39 M (0.15 M soil) | identical, all 15 | **100 %** | t2m 0.031 °C · skt 0.033 °C · sp 0.42 hPa · tau_x 0.00046 N/m² · tau_x_std 0.00048 · u10 0.0078 m/s · log_prate 0.0022 · lhtfl 0.24 W/m² |
| `occci025d` | 2.59 M | identical | **100 %** | log_chl 0.0014 · chl_cov 0.00036 |

The same check on the frames BEFORE their float16 storage (the derivation
alone) is 100 % within tolerance for 23 of 24 channels; `lhtfl` has **one**
cell of 390,960 over by 1.4 × 10⁻⁶ W/m² — float32 cancellation in a near-zero
mean. Every stored frame read back bit-identical to the float16 of what was
written. `tau_x_std`/`tau_y_std` reproduce at frame 2 to within tolerance in
every cell, which is the D3 identity measured on real data.

The same rule runs offline in `tests/test_family7_daily.py` against the REAL
family-7 builder's `glorys`, `sst`, `ncep` and `occci` stages on their own
synthetic smoke sources, compared before the z-score (10 tests, ~55 s, no
network).

**The rule for the build** (§7): every fetch lane runs this falsifier on every
bin it writes — one range read of the pentad slab per bin and group, ≈ 1 GB a
year for a 0.25° store — and **refuses to mark a year that fails**. That puts
the guard where the inputs are all it has cost (ml/CLAUDE.md §0.3).

## 5 · The probe's other numbers

### 5.1 Bytes and validity, January 2015 (zstandard 0.25.0 here; the workflow pins 0.23.0 — sizes carry over, exact shard bytes do not, as E-085 measured)

| store | stored bytes / frame | raw float16 / frame | compression | valid fraction (of all cells) |
|---|---|---|---|---|
| `glorys025d` | **6.180 MB** | 10.38 MB | 1.68× | 66.8 % (ocean north of 80° S) |
| `oisst025d` | **0.893 MB** | 4.15 MB | 4.65× | sst 67.8 %, sea_ice 14.4 % |
| `ncep100d` | **1.531 MB** | 1.95 MB | 1.28× | 100 % (soil 38.2 %) |
| `occci025d` | **0.607 MB** | 4.15 MB | 6.85× | 19.6 % |

Each bin's tile index is 1,568 bytes.

### 5.2 Wall time and memory (this sandbox: 2 vCPU, 7 GB)

Derive + write, per frame: GLORYS 0.8–1.1 s, OISST 1.2 s, NCEP 0.43 s,
OC-CCI 2.0 s (the 4 km block mean dominates). zstd level 15 is most of the
write. The whole four-store probe ran in 4 min 7 s with a peak RSS of 4.25 GB
(the 4 km OC-CCI field in float64). Downloads here: a GLORYS chunk 233–257 MB
in 5–12 s from the Hub; an OISST year 476 MB in 11 s and the 13 NCEP files of a
year in 16 s from PSL; 30 OC-CCI days (2.41 GB) in 34 s from CEDA.

### 5.3 Physical versus z-scored float16 (D4) — largest rounding error over the month

| channel | physical | z-scored with the pentad norm | ÷ day-to-day sd (physical) |
|---|---|---|---|
| sst | 0.015 °C | 0.006 °C | 7.1 % |
| sp | 0.50 hPa | 0.19 hPa | 8.9 % |
| tsoil | 0.031 °C | 0.016 °C | 2.5 % |
| ssh | 0.0005 m | 0.0007 m | 2.2 % |
| cur_u | 0.0010 m/s | 0.0007 m/s | 1.7 % |
| log_swe | 0.0039 | 0.0035 | 1.8 % |
| t2m | 0.016 °C | 0.022 °C | 0.6 % |
| every other channel | — | — | ≤ 1.6 % |

Z-scoring is better for `sst` and `sp` (≥ 512 the float16 step is 0.5, ≥ 1024
it is 1.0 hPa) and no better or worse for the rest. Both errors are far below
OISST's (~0.3 °C) and NCEP's (~1 hPa) own uncertainty, hence D4.

## 6 · Storage arithmetic, full record

| store | frames | stored (measured MB/frame × frames) | raw float16 | parts per year |
|---|---|---|---|---|
| `glorys025d` | 11,688 (1993–2024) | **72.2 GB** | 121.4 GB | 2.26 GB |
| `oisst025d` | 15,706 (1982–2024) | **14.0 GB** | 65.2 GB | 0.33 GB |
| `ncep100d` | 15,706 | **24.0 GB** | 30.7 GB | 0.56 GB |
| `occci025d` | 9,981 (1997-09-04 → 2024) | **≈ 6.7 GB** | 41.5 GB | ≈ 0.25 GB |
| **total** | | **≈ 117 GB** | **258.7 GB** | |

GLORYS and NCEP do not change coverage with the season, so their totals are
solid. OISST's `sea_ice` moves with the season (± a few % of the store).
OC-CCI's coverage changes with season and with the satellite era: the oc4k
store's bytes per frame, per year, divided by its own January-2015 figure,
average **1.11** over 1997–2022 (0.45 in 1997, SeaWiFS alone, to 1.86 in 2019),
which is the factor applied. For comparison: the pentad tensor 7.2 is 61.2 GB
for one fifth of the frames; family 5, the North Atlantic daily tensor, was
165.6 GB dense for a window 1/7 of the globe.

## 7 · Lanes, assembly, cost and wall time

**The hosted runner, as measured yesterday (E-085 canary):** 85 GB free, 4
vCPU, 15 GiB RAM, 6 h; push of a lane's parts ~120 s per GB (upload, download
back, hash, `done.json` last). Every lane is an unnamed whole-years lane of
`family1-build.yml` with `--push-parts` (parts at
`partials/family7_2d/<store>/<year>/`), on `ubuntu-latest`, no secret beyond
`HF_TOKEN` for the push.

| store | lanes | per lane | disk peak in a lane |
|---|---|---|---|
| `glorys025d` | **8 × 4 years** (1993–96 … 2021–24) | ~12 GB of chunks from the Hub; ~1,461 frames × ~1 s ≈ 25–30 min; the falsifier ≈ 4 GB of pentad range reads; ~9 GB of parts ≈ 18 min to push → **≈ 1 h** | ~10 GB |
| `oisst025d` | **9 × 5 years** (2022–24 the last) | 5 × 0.55 GB in; ~1,826 × 1.2 s ≈ 38 min; 1.6 GB of parts → **≈ 45 min** | ~3 GB |
| `ncep100d` | **9 × 5 years** | 5 × 0.5 GB in (+ the neighbouring years' `uflx`/`vflx`, 2 × 75 MB, for the centred σ at 1 Jan and 31 Dec); ~1,826 × 0.43 s ≈ 13 min; 2.8 GB → **≈ 25 min** | ~4 GB |
| `occci025d` | **28 × 1 year** (1997 … 2024) | ~29 GB from CEDA ≈ **1 h a year at 12 connections** (E-077's measured `occci-partials` rate; today's sandbox rate was ~70 MB/s on 4) or PML's per-day subset for 2023–24 (E-077: 7.5–9.6 s a day); ~12 min of block means; ~0.25 GB → **≈ 1.3 h** | ~3 GB (8 days in flight) |

**54 lanes, ≈ 50 lane-hours, $0**, ≤ 12 in flight through the queue keeper
(`family1-queue.yml`) → **≈ 4.5–5 h of wall time.**

**Assembly — one rented, verified box, the stores one after another** (E-085
§6 step 2, unchanged): `--parts-from-hub --assemble streaming`, then `check`,
then `free_cache`. Box: verified host, **≥ 200 GB disk** (`glorys025d`'s
72 GB of parts hard-linked into the store, plus restore scratch and the
environment), ≥ 16 GB RAM, ≥ 1 Gbps down, ≥ 300 Mbps up, no GPU. The three
small stores (≤ 24 GB) would fit a hosted runner's 85 GB; `glorys025d` does
not with margin, so one box does all four. **Estimate 1.5–3 h, ≈ $0.6–1.5**
(scaled from E-085's 349 GB for $1.76 by bytes).

**Total: ≈ 117 GB on the public Hub, ≈ $1, ≈ 8 h end to end** (lanes ~5 h +
box ~3 h), against E-085's 349 GB for $1.76.

## 8 · What is not feasible, or not worth it

- **A daily `rg100`.** Roemmich–Gilson is a monthly product: a daily store
  would be five copies or 29 empty frames out of 30.
- **Re-fetching GLORYS from Copernicus Marine.** The 0.25° dailies are parked
  and complete; the producer path is 840 GB at 1/12° and credentialed.
- **A NEW daily z-score norm.** It would need a full pass over every frame
  before the first frame could be encoded — sequencing that breaks the
  lanes' independence — and would make daily z-values incomparable with the
  pentad's. If Chris wants z-scores, use the pentad's published norm (Q2).
- **A dense bin-major daily `.npy`** like the pentad files: 259 GB raw, a
  14.5 MB range read for every day of g025, no skipping of land — the family-5
  experience. The sharded layout is 2.2× smaller and tile-addressable.
- **A second 4 km colour store.** `oc4k` already holds the 4 km dailies;
  what is new is 0.25°.
- **Deriving `occci025d` from `oc4k`** (176 GB of Hub reads instead of 790 GB
  from CEDA): cheaper, but `oc4k` stores log₁₀ in float16, drops values
  ≤ 10⁻⁴ mg/m³ and outside its bounds, and stops at 2022-12-31 — so the
  rebuilt pentad would not match 7.2's `oc025` to its rounding. Unmeasured;
  recorded as the fallback if CEDA is slow on the day.
- **Family 5's daily wind** is not a precedent to copy:
  `build_family4.fill_wind_daily` overwrites each day's entry once per
  6-hourly sample (`fields.setdefault(d, …)[ci] = …` inside the per-sample
  loop), so its "daily" stress is the **18 UTC instant** and its σ is over
  five 18 UTC instants (read from the code, not checked against family 5's
  bytes). `ncep_daily` here averages the four samples explicitly, and a test
  pins it.

## 9 · The dispatch sequence for the follow-up

1. **Adapters** (no dispatch): four `GridAdapter`s,
   `ml/family1/adapters/{glorys025d,oisst025d,ncep100d,occci025d}.py`, each a
   thin wrapper whose `fetch_frames` gets a lane-year's inputs (Hub chunks /
   Hub PSL mirror with PSL fallback / CEDA + PML with E-077's 8–12-worker
   prefetch) and yields `ml/family7_daily.py`'s frames; family code `72d` in
   `ml/family1/adapters/__init__.py` FAMILIES and in
   `ml/build_family1_registry.py`; the per-bin falsifier as the lane's last
   step before a year is marked; a smoke per store built on
   `build_family7.make_smoke_sources` (tests/test_family1_<store>.py).
2. **Framework probes in the sandbox**:
   `python3 ml/build_family1_stores.py --store <s> --stage probe --probe-month 2015-01`
   for each store; the bytes per frame must match §5.1.
3. **Four canary lanes** on `family1-build.yml`, one per store, e.g.
   `{"store":"ncep100d","stage":"index,fetch","start":"2015-01-01","end":"2015-12-31","runner":"ubuntu-latest","extra_args":"--push-parts"}`;
   read `df`, the rates and the falsifier lines in each log before the rest.
4. **The other 50 lanes** onto the `f1-queue` branch's `lane_queue.json`
   (≤ 12 in flight, `retries: 2`).
5. **One verified box**, per store:
   `{"store":"S","stage":"all","start":"<first day>","end":"2024-12-31","runner":"gpu-box-<offer>","extra_args":"--parts-from-hub --assemble streaming"}`,
   then `{"store":"S","stage":"check",…}`, then `free_cache` on the next
   store's dispatch; destroy the box.
6. **Registry and Data tab**: `family1-registry.yml` publishes
   `tensors/family7_2d/family72d.json`; one entry in `F1Data.DEFAULT_REGISTRIES`
   (the tab already reads sharded stores, physical units, and five daily
   frames per bin); docs.

## 10 · Open questions for Chris

- **Q1 — the name.** "Family 7.2d" under `tensors/family7_2d/`, listed in the
  Data tab beside the pentad and the climatology? (Recommended.)
- **Q2 — physical or z-scored.** Physical (D4, recommended) or z-scored with
  the pentad's norm (2.7× finer for `sst` and `sp`, readers multiply back)?
- **Q3 — the daily storminess.** Centred five-day σ (D3, recommended, keeps
  the pentad identity) or the σ of the day's four 6-hourly samples (a
  sub-daily storminess, no identity)?
- **Q4 — the record end.** Stop at 2024-12-31 with 7.2 (recommended for now),
  or extend into 2025 (OISST, NCEP and OC-CCI continue; the GLORYS chunks
  would need a CMEMS pull of 2025)?
- **Q5 — daily truth.** Ship the RAPID and Florida Current transports as
  daily series beside the stores (family 5 had a `truth_daily.npz`)?

## 11 · What is NOT verified

- 382 of the 384 GLORYS chunks were listed, not opened (sizes 233–262 MB are
  consistent, and the 7.2 build consumed all of them: `n_glorys_bins` 2,339 =
  bins 803–3141). The GLORYS interim stream's seam (chunks from 2021-07 may
  carry `source_dataset` …`myint`…) was not inspected.
- The full-record sizes extrapolate one January; OISST's sea-ice season and
  OC-CCI's era are scaled, not measured (§6).
- Lane rates are this sandbox's (2 vCPU) and E-085/E-077's runner
  measurements, not a lane of these stores.
- zstandard 0.25.0 here vs the pinned 0.23.0 (sizes carry over, bytes do not);
  the probe ran from an uncommitted tree on top of `5519fc3`, which is the
  `builder_git_sha` the report records.
- Family 5's daily wind being the 18 UTC instant is read from code (§8).

## 12 · Files

- `ml/family7_daily.py` — the daily derivation per source, the pentad
  reconstruction rule, the Hub pentad reader, the probe CLI
  (`python3 ml/family7_daily.py probe --src <dir> --out <dir> --bins 2411-2416 --report <json>`).
- `tests/test_family7_daily.py` — the rules, the builder comparison on the
  smoke sources, NCEP's 6-hourly day mean and frame-2 σ, the writer round trip.
- `ml/plans/E087_probe_2015-01.json` — every number in §4–§5.

## 13 · Status — the build (2026-10-06, live)

**Decided by the planning session the same day:** Q1 "family 7.2d",
`tensors/family7_2d/`, registry `family72d.json`, listed in the Data tab;
Q2 physical units; Q3 the centred five-day σ, named as such in the channel
units; Q4 record to 2024-12-31; Q5 no daily transport series. OC-CCI is
built LAST and parked if its lanes are slow or flaky.

**Code** (`dc3f277`, `fb8c0c1`): adapters
`ml/family1/adapters/{_f7d,glorys025d,oisst025d,ncep100d,occci025d}.py`,
family code `72d`, the registry builder's family 7.2d entry, the Hub
read-back checker `ml/family1/f7d_hub_check.py`, the queue helper
`scripts/family1_enqueue.mjs`. Every fetch lane runs §4's falsifier on every
bin it writes and refuses a year that fails it.

**What the lanes' check found** (the first canaries were refused, and every
refusal was a real mechanism, now reproduced exactly — details in
`ml/family1/BUILD_LOG.md` E-087):

- NCEP's Antarctic winter skin temperature goes to −109.9 °C, below the
  first sanity bound (−100 °C); bounds are now −150 °C.
- Some NCEP years store a dry cell as −2.3 × 10⁻¹⁰, and `log1p_channel`
  clamps per day here, per pentad in 7.2: the check allows exactly the
  measured |x| × 86400 (2.0 × 10⁻⁵ in log1p mm/day).
- GLORYS reports a finite mixed-layer depth ≤ 0 on a few cell-days, which
  7.2 counts as a zero in its pentad mean; the check rebuilds `log_mld` the
  same way. For a reader: **where `log_mld` is NaN on a day inside an ocean
  bin, 7.2's pentad may have averaged that day in as a zero depth.**

**Built, published, checked, registered (2026-10-06).** Three stores, each
lane refusing any year whose bins do not rebuild the published pentad
(none refused after `fb8c0c1`), assembled and published on one verified
Vast box (54469903, 1 h 54 m, ≈ $0.27, destroyed and confirmed gone), each
followed by the framework's full `check` stage and by an independent Hub
read-back (`ml/family1/f7d_hub_check.py`):

| store | days | bytes on the Hub | files | read-back (two days) |
|---|---|---|---|---|
| `glorys025d` | 11,688 (1993-01-01…2024-12-31) | 72.36 GB | 4,682 | ≤ 0.00097 against the GLORYS chunks; pentad ok |
| `oisst025d` | 15,706 (1982-01-01…2024-12-31) | 14.20 GB | 6,288 | `sst` ≤ 0.0138 °C against PSL; pentad ok |
| `ncep100d` | 15,706 (1982-01-01…2024-12-31) | 23.96 GB | 6,288 | `t2m` ≤ 0.031 °C, `sp` ≤ 0.50 hPa against PSL; pentad ok |

Registry:
[family72d.json](https://huggingface.co/datasets/chfrank/earth-tensors/blob/main/tensors/family7_2d/family72d.json)
(4 stores, 3 built, `occci025d` listed as not built). Every timing, margin
and box number: `ml/family1/BUILD_LOG.md`, E-087.

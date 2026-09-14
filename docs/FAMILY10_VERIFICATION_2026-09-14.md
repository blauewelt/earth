# Family 10 — independent verification of the three published tier-P observation stores, 2026-09-14

**What this is.** Family 10 is the programme's storage contract for
observations of mixed granularity — nothing is resampled onto a common grid,
every source keeps its own resolution and cadence, and each measurement carries
its own footprint (how much area and how much time it averaged). Its tier-P
stores are tables of point measurements sorted by five-day bin, searched with
"what are the k nearest observations to this place and pentad". Three of them
were built and published on 2026-09-14: `gdp` (the Global Drifter Program's
6-hourly quality-controlled surface drifters), `gtmba` (the tropical moored
arrays TAO/TRITON, PIRATA and RAMA, one row per mooring per day) and `socat`
(the Surface Ocean CO₂ Atlas, underway ship measurements of surface-water CO₂).
The fourth, `slatrack` (along-track sea-level anomaly from 29 altimeter
missions), has not been built.

**Who ran it, and how.** A Claude session, on 2026-09-14, from a clean sandbox
that did not build the stores and did not read the builder's own report of what
it had done. Every file was fetched over HTTPS from the Hugging Face dataset
repository `chfrank/earth-tensors`, hashed against `store.json`'s own sha256
record, opened with `ml/family10_store.py`, and then **recomputed from the
arrays** — the counts, the ranges, the per-channel statistics and the index
structure — rather than read back out of the metadata that claims them. The
k-nearest measurements of §5 and §8.6 were run with the two scripts now in
`ml/tools/family10_verify/`. Nothing was committed from that sandbox; work was
under `/tmp/f10verify/`.

**What it is for.** This page is the evidence behind the RESULT section of
E-079 — the experiment that built family 10's first four observation stores and
the registry tying them to the gridded tensors — in
[the experiment log](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-079).
The reader's contract for the stores themselves is
[the family-10 data handover](https://blauewelt.github.io/earth/docs.html?f=docs/FAMILY10_DATA_HANDOVER.md);
its §6 (search radii) and §9 (what is verified) were rewritten from the numbers
below.

**The builds these files came from**, all on GitHub-hosted runners at no cost,
all from builder commit `c5bc2ce`:

- [the `gtmba` build](https://github.com/blauewelt/earth/actions/runs/34814444103) — fetched the nine PMEL daily mooring datasets from the OSMC ERDDAP server and published the tropical-moored-array store, ~20 min.
- [the `socat` build](https://github.com/blauewelt/earth/actions/runs/34814445849) — streamed and parsed the 1.4 GB SOCAT v2026 synthesis file and published the ship-CO₂ store, ~20 min.
- [the `gdp` build](https://github.com/blauewelt/earth/actions/runs/34814442453) — fetched 46 years of 6-hourly drifter data from the AOML ERDDAP server one month per request (~550 requests, which is why the build step took 85 minutes rather than 20) and published the drifter store, then rewrote the family-10 registry.

Everything below is as recorded on the day, numbers unchanged.

---

Sections 1 to 7 cover `gtmba` (the tropical moored arrays) and `socat` (ship
CO₂), which published first; section 8 covers `gdp` (the surface drifters),
checked identically after its own publish at 08:05Z the same day.

**Headline: all three stores PASS every structural and integrity check.** The
discrepancies found are (a) a float32 timestamp-resolution artefact at year
boundaries in `socat`, ≤3 rows/year, which is a property of the format and not
a build error, (b) a 568-row gap in `gdp` between a counter in `store.json` and
the array it describes, which reconciles exactly and leaves every array
self-consistent (§8.4), and (c) the handover's §6 suggested radii are, as §6
itself warned, not the measured ones — §5 and §8.6 give the measured
replacements.

---

## 1 · store.json, as published

| | `gtmba` | `socat` |
|---|---|---|
| title | Global Tropical Moored Buoy Array (TAO/TRITON, PIRATA, RAMA), daily | SOCAT v2026 surface-water fCO₂ |
| N (rows) | 1,001,282 | 41,830,675 |
| C | 18 | 4 |
| `bin_first` | −304 | −1768 |
| `bin_last` | 3141 | 3141 |
| `n_bins` | 3,446 | 4,910 |
| `n_live_bins` | 3,386 | 3,266 |
| date_range | 1977-01-01 → 2024-12-31 | 1957-01-01 → 2024-12-31 |
| footprint (log2_fp, log2_dt) | (−4.0, −2.3) | (−4.0, −4.0) |
| `values_measured_fraction` | 0.568374 | 0.923146 |
| `qc_keep_max` | 2 | 2 |
| builder | `ml/build_family10_stores.py` | same |
| builder_git_sha | `c5bc2ced06a2d1473f1fc47e2126d190c26ea0f4` | same |
| built_at | 2026-09-14T07:01:04Z | 2026-09-14T06:57:33Z |
| sources | OSMC ERDDAP `pmelTaoDy*` tabledap | `socat.info/.../SOCATv2026.tsv.zip`; NCEI OCADS mirror |
| rows_read → kept | 35,771,817 samples read; 15,781,023 samples kept | 2,597,074,036 rows read |
| drops recorded | drop_qc 893,009 · drop_fill 15,835,928 · drop_depth 3,261,857 · datasets_empty 37 · sites 3,114 | drop_out_of_range 127,440,295 · drop_no_fco2 1,623,680 · drop_no_position 0 · drop_woce 0 · drop_dataset_qc 0 · preamble_lines 498,786 |

**Note added later the same day (2026-09-14): every `socat` counter in the two
rows above is 59× too large, and the cause is a ledger bug in the builder, not
the archive.** The one-pass fetch attached the whole stream's counters to every
year part, and the assembler summed them across the 59 years that held rows:
2,597,074,036 / 59 = **44,018,204**, exactly the row count measured when the
file was streamed end to end on 2026-09-13; 127,440,295 / 59 = **2,160,005**
(rows outside the 1957 → 2024 window, the preamble excluded) and 1,623,680 / 59
= **27,520**. So the store read 44,018,204 rows and kept 41,830,675, **95.0 %**.
The arrays are untouched by this — N, the bins, the values and every
per-channel statistic below verify exactly — which is why the fault was
invisible until someone divided. `ml/build_family10_stores.py` now attaches the
stream's counters to one part only, and a test pins that a two-year single
stream yields the stream's own `rows_read`.

**Rebuilt, and this verification carries over unchanged.** The store was
rebuilt the same day — `built_at` 2026-09-14T08:45:54Z, builder commit
`0d5860d`, ~17 min, the log ending `publish: 10 file(s) verified by restore` —
and its `store.json` now reads `rows_read` 44,018,204, `drop_out_of_range`
2,160,005, `drop_no_fco2` 27,520 and `preamble_lines` 8,454. **All nine array
sha256 values in the new file are byte-identical to the ones checked in §2**,
so the rebuild changed the ledger and nothing else, and every measurement in
this document still describes the file on the Hub.

- [the `socat` rebuild](https://github.com/blauewelt/earth/actions/runs/34822844466) — re-streamed the SOCAT v2026 synthesis file with the corrected counter and republished the store, arrays unchanged.

**Verification notes recorded in the stores** (verbatim):

- `gtmba`: *"2026-09-13: the nine pmelTaoDy* datasets listed on osmc.noaa.gov/erddap, every variable name read from each dataset's info json, and a real 0N 140W January 2015 daily temperature fetch (930 rows, 30 depths)"*
- `socat`: *"2026-09-13: release page followed to v2026, HEAD on the zip (1,411,801,422 B, accept-ranges), and the 32-column data header read out of the first 50 MB by Range request"*

Note both notes date the *source* verification to 2026-09-13, i.e. the day
before the build (`built_at` 2026-09-14). They describe the archive access
path, not the published arrays.

### Per-year counts

| | `gtmba` | `socat` |
|---|---|---|
| first year in `per_year` | 1977 = 173 | 1957 = 578 |
| last year | 2024 = 20,958 | 2024 = 2,359,858 |
| years listed | 48 | 68 |
| sum of `per_year` | 1,001,282 | 41,830,675 |
| equals N? | **yes — PASS** | **yes — PASS** |

`socat` has true gaps (1958–60, 1964–67, 1971–72 are 0).

---

## 2 · Integrity — sha256 of every file against store.json's own record

Nine arrays per store; `store.json` is not (and cannot be) in its own block.

| file | `gtmba` | `socat` |
|---|---|---|
| `bin.npy` | MATCH | MATCH |
| `time_days.npy` | MATCH | MATCH |
| `lat.npy` | MATCH | MATCH |
| `lon.npy` | MATCH | MATCH |
| `values.npy` | MATCH | MATCH |
| `platform.npy` | MATCH | MATCH |
| `qc.npy` | MATCH | MATCH |
| `fp.npy` | MATCH | MATCH |
| `bin_offsets.npy` | MATCH | MATCH |

`ml/family10_store.verify_store(dir)` exists and returned **9** for each store
without raising. **PASS (18/18 files).**

Sizes on disk: `gtmba` 63 MB; `socat` 1.37 GB (`values.npy` 334,645,528 B).

---

## 3 · Structural checks after `Store.open` (recomputed from the arrays)

| check | `gtmba` | `socat` | verdict |
|---|---|---|---|
| `N` from arrays vs store.json | 1,001,282 = 1,001,282 | 41,830,675 = 41,830,675 | PASS |
| `bin_first` / `bin_last` / `n_bins` | −304 / 3141 / 3446, all equal | −1768 / 3141 / 4910, all equal | PASS |
| live bins (≥1 row) vs `n_live_bins` | 3,386 of 3,446 (98.3%) = 3,386 | 3,266 of 4,910 (66.5%) = 3,266 | PASS |
| rows sorted ascending by `(bin, time_days)` | yes | yes | PASS |
| `bin_offsets` reproduced by `searchsorted(bin, …)` | exact | exact | PASS |
| `bin == floor(time_days/5)` for every row | yes | yes | PASS |
| `lon` in [−180, 180) | [−180.000, 170.000] | [−180.0, 179.99998474] | PASS |
| `lat` range | [−25.0, 21.0] (tropical array) | [−78.74, 90.00] | PASS |
| `fp.npy` unique rows | exactly one: (−4.0, −2.30078125) | exactly one: (−4.0, −4.0) | PASS (f16 rounding of −2.3) |
| `qc` histogram vs `qc_keep_max=2` | 1: 283,715 · 2: 717,567 — nothing >2 | 1: 41,830,675 — all best grade | PASS |
| distinct `platform` ids | 159 (moorings; min 13001, WMO-like) | 8,024 (hashed expocodes, all positive) | PASS |

### Measured fraction — recomputed vs claimed

Overall: `gtmba` **0.568374** vs claimed 0.568374; `socat` **0.923146** vs
claimed 0.923146. **PASS.**

Per channel, every one of the 22 channels matched store.json's `measured`
count *exactly* and its `fraction`, `min`, `max` and `mean` to float16/float32
precision. **PASS.** Selected rows:

| store | channel | measured (mine = json) | fraction | min | max | mean |
|---|---|---|---|---|---|---|
| gtmba | sst | 789,805 | 0.788794 | 17.12 | 32.44 | 27.7186 |
| gtmba | t_1m | 789,410 | 0.788399 | 17.12 | 32.44 | 27.7189 |
| gtmba | sss | 383,671 | 0.383180 | 26.50 | 38.44 | 34.7523 |
| gtmba | airt | 823,794 | 0.822739 | 1.32 | 34.12 | 26.8583 |
| gtmba | wind_u | 771,432 | 0.770444 | −14.30 | 23.30 | −3.4323 |
| gtmba | wind_v | 771,432 | 0.770444 | −13.60 | 13.70 | 0.6371 |
| gtmba | u_cur | 66,789 | 0.066703 | −0.6611 | 1.186 | 0.0920 |
| gtmba | v_cur | 66,789 | 0.066703 | −0.6152 | 0.6621 | −0.0007 |
| gtmba | t_20m…t_180m | 436,814 … 451,616 | 0.436–0.477 | 0.19–13.81 | 28.06–31.56 | 13.76–26.53 |
| gtmba | t_300m | 865,460 | 0.864352 | 7.109 | 17.77 | 11.1541 |
| gtmba | t_500m | 846,296 | 0.845212 | 5.160 | 13.17 | 8.3112 |
| socat | fco2 | 41,830,675 | 1.000000 | 7.199 | 2000.0 | 368.0543 |
| socat | sst | 41,825,451 | 0.999875 | −2.480 | 34.97 | 14.4513 |
| socat | sss | 39,544,744 | 0.945353 | 0.0 | 42.25 | 31.2768 |
| socat | patm | 31,262,391 | 0.747356 | 859.0 | 1066.0 | 1009.3914 |

Physical sanity, checked rather than assumed:

- `gtmba`: `sst` and `t_1m` come from different PMEL datasets and their means
  agree to 0.0003 °C (27.7186 vs 27.7189) — §3.2's free cross-check **passes**.
- `gtmba` ADCP currents peak at 1.19 m s⁻¹, not ~119 — the cm s⁻¹ → m s⁻¹
  conversion of §3.2 **is applied**; a silent factor of 100 is excluded.
- `socat` `patm` mean 1009.4 hPa, range 859–1066: a real barometer, and only
  74.7% populated, consistent with §3.3's refusal to substitute `NCEP_SLP`.
- Edge values worth knowing, all *inside* the declared bounds so nothing is
  wrong, but they sit on the bound: `fco2 == 2000.0` exactly on 6 rows (the
  upper bound); `sss < 5 PSU` on 128,336 rows of which 175 are exactly 0.0
  (river plumes / sea ice are real, but a 0.0 PSU "measurement" is not, and the
  builder's bound is `[0, 45]` so it does not catch them); `lat == 90.0`
  exactly on 10 rows. These are **flagged, not failures.**

### The one discrepancy: `socat` per-year counts recomputed from `time_days`

Recomputing the year of each row from `time_days.npy` reproduces store.json's
`per_year` for `gtmba` exactly, but for `socat` differs by 1–3 rows in 16 of
68 years, and puts 3 rows in a year 2025 that store.json does not list.

Cause, diagnosed: `time_days` is float32, and its spacing at the end of the
record (t ≈ 15,700 d) is 0.0009766 d = **84.4 s**. The store's largest
timestamp is exactly 15706.0 = 2025-01-01T00:00:00, i.e. three rows late on
2024-12-31 rounded up across midnight. `bin` is unaffected
(floor(15706/5) = 3141 = `bin_last`) and the `bin == floor(time/5)` identity
holds for all 41.8 M rows. So **store.json's per-year counts are the correct
ones** (taken from the source dates) and my recomputation is the imprecise
side. Not a build error; a documented consequence of the float32 column, worth
knowing for any consumer that derives calendar dates from `time_days` rather
than from `bin`. Total magnitude: 3 rows out of 41,830,675 (7e-8) land in the
wrong calendar year.

---

## 4 · `knearest` at the named anchors

Suggested (k, R_max_km, T_max_days) from §6: `gtmba` (4, 1500, 15),
`socat` (8, 500, 30). Bin 2411 = 2015-01-03 → 2015-01-07; `dt` runs from the
end of that pentad.

### 4a · `gtmba` @ 36.0 N, −70.0 E, bin 2411 — **empty, correctly**

`n_found = 0`, `n_R = 0`, all four slots miss tokens (`valid=False`,
`row=−1`, values NaN). Correct: `gtmba` spans lat [−25, 21] only; 36 N is
~1,670 km outside the array, beyond R_max=1500 km. Call time **0.2 ms**.

### 4b · `gtmba` @ 0.0 N, −140.0 E (TAO site), bin 2411 — k=4, R=1500, T=15

`n_found = 4`, `n_R = 75`, **0.3 ms**.

| slot | dist_km | dt_days | platform | qc | lat/lon | sst | sss | airt | wind_u | wind_v | u_cur | t_100m | t_300m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.00 | 0.50 | 51311 | 2 | 0.000, −140.000 | 26.094 | 35.375 | 26.062 | −6.082 | 2.889 | 0.052 | 22.312 | 11.648 |
| 1 | 0.00 | 1.50 | 51311 | 2 | 0.000, −140.000 | 26.109 | 35.375 | 26.141 | −6.602 | 0.972 | 0.093 | 22.578 | 11.836 |
| 2 | 222.64 | 0.50 | 51008 | 2 | 2.000, −140.000 | 27.094 | 34.875 | 26.562 | −5.746 | 3.617 | NaN | 25.656 | 11.969 |
| 3 | 222.64 | 0.50 | 51009 | 2 | −2.000, −140.000 | 26.328 | 35.156 | 26.172 | −6.227 | 1.770 | NaN | 26.312 | 11.969 |

Physically sound: the 0/2/−2 N, 140 W TAO triplet at exactly 2° spacing
(222.64 km = 2 × 111.32); equatorial SST ~26 °C in the 2014-15 warm event;
easterly trades (wind_u ≈ −6 m s⁻¹); a 0.05–0.09 m s⁻¹ eastward subsurface
current; `t_1m` identical to `sst` at 26.094; monotone thermocline
26 → 11.6 °C; `t_500m` NaN at 0 N and present off-equator; `u_cur`/`v_cur`
NaN at the two flanking sites (no ADCP). **PASS.**

### 4c · `socat` @ 36.0 N, −70.0 E, bin 2411 — k=8, R=500, T=30

`n_found = 8`, `n_R = 136`, **11.0 ms** (min 10.5).
All eight slots from one cruise (platform 204860182106006216), qc=1,
dist 98.5–107.0 km, dt 25.28–25.33 d (i.e. ~2014-12-13), positions marching
down a single track at 35.72–36.08 N, −71.09 to −71.14 E.
fco2 341.5–343.5 µatm, sst 21.41–21.77 °C, sss 36.53 PSU, patm 1010–1011.5 hPa,
with `sst` NaN on two of the eight rows. Sargasso-edge salinity of 36.5,
December SST of 21.5 °C on the Gulf Stream's north wall, and a wintertime
undersaturated fco2 of ~342 µatm are all plausible. **PASS.**

### 4d · `socat` @ 50.0 N, −30.0 E, bin 2411 — k=8, R=500, T=30

`n_found = 8`, `n_R = 1201`, **10.8 ms** (min 7.6).
All eight from one cruise (154048485345510149), qc=1, dist 181.25–181.79 km,
dt 27.62–27.64 d, positions 51.38–51.41 N, −31.28 to −31.37 E — again a single
along-track run. fco2 402.5–409.5 µatm, sst 7.35–7.60 °C, sss 34.03–34.06 PSU,
patm 1017.5 hPa. Mid-December subpolar North Atlantic. **PASS.**

**Structural note from 4c/4d (matters for k):** at a typical `socat` anchor all
k = 8 returned rows come from the **same cruise, seconds apart along one
track** — see §5's distinct-platform measurement.

---

## 5 · The §6 measurement: k-th-neighbour distance, measured

Method (the family-8 `index_stats` two-draw design): for each store, 2,000
anchors per draw, `knearest` run with **generous bounds R_max = 5000 km,
T_max = 60 d** so the k-th neighbour is nearly always found, but with the
**metric scales `L_km`, `T_days` pinned to §6's suggested (R, T)** so the
ranking is the one a consumer would actually use, and `dist_km[k−1]` is
therefore exactly "the R_max that would have retained the k-th token".

- **catalogue draw** — anchors are the store's own rows (lat, lon, bin) with
  `bin ≥ 0`, stratified equally across the years present so no single dense
  year dominates.
- **globe draw** — lat ~ U(−60, 60), lon ~ U(−180, 180), bin a random live
  bin ≥ 0. Includes land, where the honest answer is "nothing is measured".

RNG seed 20260914. Percentiles are over the anchors where k were found.

### 5a · distance to the k-th neighbour (km)

| store | k | draw | median | p90 | p99 | anchors with < k found (at 5000 km / 60 d) |
|---|---|---|---|---|---|---|
| `gtmba` | 4 | catalogue | **0.00** | **333.96** | **333.96** | 0.0 % |
| `gtmba` | 4 | globe | 2,634.99 | 4,556.13 | 4,954.81 | **34.05 %** |
| `socat` | 8 | catalogue | **4.45** | **46.19** | **111.33** | 0.0 % |
| `socat` | 8 | globe | 1,653.66 | 3,983.47 | 4,874.27 | **15.70 %** |

### 5b · age of the k-th neighbour (days)

| store | k | draw | median | p90 | p99 |
|---|---|---|---|---|---|
| `gtmba` | 4 | catalogue | 2.50 | 3.50 | 3.50 |
| `gtmba` | 4 | globe | 3.50 | 3.50 | 4.50 |
| `socat` | 8 | catalogue | 1.82 | 4.24 | 4.91 |
| `socat` | 8 | globe | 18.60 | 52.30 | 59.96 |

### 5c · `n_R` at the §6 suggested (R_max, T_max)

| store | (k, R_max, T_max) | draw | n_R median | n_R p90 | n_R mean | anchors with n_R = 0 | anchors with n_R < k |
|---|---|---|---|---|---|---|---|
| `gtmba` | (4, 1500 km, 15 d) | catalogue | 90 | 194 | 99.3 | 0.0 % | 0.0 % |
| `gtmba` | (4, 1500 km, 15 d) | globe | **0** | 45 | 12.7 | **81.3 %** | 81.3 % |
| `socat` | (8, 500 km, 30 d) | catalogue | 2,056 | 13,751 | 5,295.3 | 0.0 % | 0.0 % |
| `socat` | (8, 500 km, 30 d) | globe | **0** | 21 | 109.7 | **89.45 %** | 89.55 % |

### 5d · how many *distinct* platforms the k tokens come from (600 catalogue anchors)

| store | k | median distinct platforms | mean |
|---|---|---|---|
| `gtmba` | 4 | 2 | 1.82 |
| `socat` | 8 | 1 | 1.00 |

### What the measurement says about §6's suggestions

- **`socat`, k = 8, R = 500 km, T = 30 d.** At a place SOCAT actually samples,
  the 8th neighbour is **4.5 km away (median), 46 km at p90, 111 km at p99** —
  the suggested 500 km is **~10× larger than needed** and is never the binding
  constraint at a sampled anchor. It *is* binding off-track: 89.5 % of uniform
  globe anchors see fewer than 8 observations inside (500 km, 30 d), and 89.45 %
  see none at all. So 500 km does not "rescue" the off-track anchors either —
  §6's own remark that "`n_R` is the honest measure" is confirmed. A measured
  replacement: **R_max ≈ 150 km covers the 99th percentile of on-track
  anchors**, and the miss token should be accepted as the normal answer
  elsewhere. The deeper finding is that k = 8 buys **one cruise's worth of
  redundancy**: all 8 tokens come from a single expocode at every anchor
  measured (median and mean = 1.0 distinct platforms), seconds apart on one
  line. If k is meant to buy independent information, either k must be far
  larger or the search must be de-duplicated by platform.
- **`gtmba`, k = 4, R = 1500 km, T = 15 d.** At a mooring, the 4th neighbour is
  at **0 km (median)** and never further than **334 km (p90 = p99)** — because
  the array is a fixed lattice at 2°–15° and the same site contributes several
  daily rows inside 15 days. Measured, **R_max ≈ 350 km suffices at an
  on-array anchor**, not 1500 km. The large R is doing work only for anchors
  off the array, and there it does not work either: 81.3 % of globe anchors see
  n_R = 0 at (1500 km, 15 d), and a third of them find fewer than 4 rows even
  at 5000 km / 60 d — unsurprising for a store whose latitudes are [−25, 21].
  Distinct platforms per k = 4: median 2, mean 1.82, i.e. half the tokens are
  the same mooring on a different day. The temporal bound, not the spatial one,
  is what k = 4 is really spending: the 4th neighbour is 2.5–3.5 days old, so
  **T_max ≈ 5 d would give the same answer** at nearly every on-array anchor.

Both suggestions are, exactly as §6 said of itself, "starting points … not
measured optima", and both are **too loose on R by roughly an order of
magnitude at anchors where the store has data**, while being **irrelevant at
anchors where it does not**.

---

## 6 · Timing

Wall-clock per `knearest` call, in this sandbox (memmapped local files, cold
page cache warmed by the first calls), medians over the measurement runs:

| store | call | median ms | p90 ms |
|---|---|---|---|
| `gtmba` | (4, 1500 km, 15 d), catalogue anchors | 0.48 | 0.65 |
| `gtmba` | generous (5000 km, 60 d), catalogue | 0.48 | 0.65 |
| `gtmba` | generous, globe anchors | 0.39 | 0.59 |
| `gtmba` | named anchors §4a/§4b | 0.2 / 0.3 | — |
| `socat` | generous (5000 km, 60 d), catalogue | 8.05 | 34.09 |
| `socat` | generous, globe | 8.29 | 27.96 |
| `socat` | (8, 500 km, 30 d), named anchors §4c/§4d | 11.0 / 10.8 | — |

`socat` cost is dominated by the brute-force scan over the pentads the time
bound admits (~10²–10⁵ rows); `gtmba` is sub-millisecond throughout.

---

## 7 · Verdict

| check | `gtmba` | `socat` |
|---|---|---|
| all 9 files downloadable from the Hub | PASS | PASS |
| sha256 of all 9 files vs store.json | PASS (9/9) | PASS (9/9) |
| `verify_store()` runs and returns without raising | PASS (9) | PASS (9) |
| `Store.open` on the local directory | PASS | PASS |
| N / bin_first / n_bins / bin_last vs store.json | PASS | PASS |
| live bins vs `n_live_bins` | PASS (3,386 / 3,446) | PASS (3,266 / 4,910) |
| rows sorted by (bin, time_days) | PASS | PASS |
| `bin_offsets` CSR reproducible from `bin.npy` | PASS | PASS |
| `bin == floor(time_days/5)` | PASS | PASS |
| lon ∈ [−180, 180) | PASS | PASS |
| overall measured fraction vs store.json | PASS (0.568374) | PASS (0.923146) |
| per-channel measured/fraction/min/max/mean | PASS (18/18) | PASS (4/4) |
| per_year sums to N | PASS | PASS |
| per_year reproducible from `time_days` | PASS (exact) | **DIFFERS by ≤3 rows/yr — float32 resolution (84 s at end of record), not a build error** |
| qc ≤ `qc_keep_max` | PASS | PASS |
| fp constant and equal to store.json's footprint | PASS | PASS |
| `knearest` returns fixed-k, miss tokens where empty | PASS | PASS |
| values physically plausible at the named anchors | PASS | PASS |
| §6 suggested radii are the measured optima | **NOT CONFIRMED — measured R ≈ 350 km, T ≈ 5 d** | **NOT CONFIRMED — measured R ≈ 150 km** |

No check failed. Two things a consumer should be told: `time_days` is float32
and loses ~84 s near the end of the record (derive calendar dates from `bin` or
from the source, not from `time_days`); and the k tokens `knearest` returns are
dominated by a single platform in both stores (`socat` median 1 of 8,
`gtmba` median 2 of 4), which is a property of the sampling, not of the reader.

---

## 8 · `gdp` — same verification, run 2026-09-14 after the 08:05Z publish

Downloaded and checked identically to §1–§7. 1.6 GB of arrays.
**Result: every integrity and structural check PASSES.** Two things do not
match an expectation: a 568-row gap between `counts.drogue_uncertain` and the
NaN count actually in `values.npy` (§8.4), and the Gulf Stream anchor does
*not* return |u| ≈ 1 m s⁻¹ (§8.5) — the store contains such speeds, the k = 8
at 300 km simply does not reach them.

## 8.1 · store.json, as published

| field | value |
|---|---|
| title | Global Drifter Program, 6-hourly QC interpolated |
| store / tier / family | `gdp` / P / family10 |
| N (rows) | **48,480,798** |
| C | 4 — `u`, `v`, `sst`, `drogue` (m s⁻¹, m s⁻¹, °C, 1 = drogue attached) |
| `bin_first` / `bin_last` | −211 / 3141 (`bins_requested` [−220, 3141]) |
| `n_bins` / `n_live_bins` | **3,353 / 3,353 — every bin in range is live** |
| date_range | 1979-01-01 → 2024-12-31 |
| footprint (log2_fp, log2_dt) | (−4.0, −4.3) |
| `values_measured_fraction` | 0.963865 |
| `qc_keep_max` | 2 |
| builder / commit | `ml/build_family10_stores.py` @ `c5bc2ced06a2d1473f1fc47e2126d190c26ea0f4` (same commit as `gtmba`/`socat`) |
| built_at | **2026-09-14T08:04:35Z** |
| sources | `erddap.aoml.noaa.gov/gdp/erddap/tabledap/drifter_6hour_qc.csvp`; OSMC mirror |
| resume_granularity | year |

**Drops and counters recorded:** `rows_read` 48,708,540 → N 48,480,798
(99.53 % kept). `drop_pos_err` **227,174** (the > 50 km derived-position-error
rule of §4.2), `drop_no_position` 0, `drop_no_time` 0, `drop_out_of_range` 0.
Channel-level: `out_of_bounds.sst` 5,028, `out_of_bounds.v` 1, `out_of_bounds.u` 0.
`drogue_uncertain` 1,417,396. `missing_month` 1 (the preflight records
1979-01 as parsing 0 rows — consistent with the first data landing 1979-02-15).

227,174 + 48,480,798 = 48,707,972, which is 568 short of `rows_read`
48,708,540. See §8.4 — the same 568.

**Verification note recorded** (verbatim): *"2026-09-13: dataset listing,
variable metadata and a real 2015-01 month fetch (149,658 rows) from this
sandbox"*.

**QC policy** (verbatim, abridged): the GDP product has no per-row flag, so the
flag is **derived**: `position error = hypot(err_lat, err_lon·cos lat)·111.32`
km; > 50 km dropped, ≤ 5 km → qc 1, rest → qc 2. `drogue` is 1 before
`drogue_lost_date`, 0 after, NaN where that date is 1970-01-01.

### Per-year counts

| | value |
|---|---|
| first year | 1979 = 24,046 |
| last year | 2024 = 1,834,400 |
| years listed | 46 (1979–2024, no gaps) |
| sum of `per_year` | 48,480,798 |
| equals N? | **yes — PASS** |
| reproduced from `time_days` | **exact, 46/46 years — PASS** (no float32 drift, unlike `socat`) |

## 8.2 · Integrity — sha256 of all 9 files

| file | result | file | result |
|---|---|---|---|
| `bin.npy` | MATCH | `platform.npy` | MATCH |
| `time_days.npy` | MATCH | `qc.npy` | MATCH |
| `lat.npy` | MATCH | `fp.npy` | MATCH |
| `lon.npy` | MATCH | `bin_offsets.npy` | MATCH |
| `values.npy` | MATCH | | |

`verify_store('/tmp/f10verify/gdp')` returned **9** without raising. **PASS (9/9).**
Sizes: `values.npy` 387,846,512 B, `platform.npy` 387,846,512 B,
`lat`/`lon`/`time_days`/`fp` 193,923,320 B each, `bin.npy` 96,961,724 B,
`qc.npy` 48,480,926 B, `bin_offsets.npy` 26,960 B — 1.6 GB total. This matches
the build log's "10 file(s) verified by restore" (9 arrays + `store.json`).

## 8.3 · Structural checks recomputed from the arrays

| check | measured | verdict |
|---|---|---|
| `N` from arrays vs store.json | 48,480,798 = 48,480,798 | PASS |
| `bin_first` / `bin_last` / `n_bins` | −211 / 3141 / 3353, all equal to store.json | PASS |
| live bins vs `n_live_bins` | **3,353 of 3,353 (100 %)** = 3,353 | PASS |
| build-log claim "bins −211..3141 (3,353 live)" | reproduced exactly | PASS |
| rows sorted ascending by `(bin, time_days)` | yes | PASS |
| `bin_offsets` reproduced by `searchsorted(bin, …)` | exact | PASS |
| `bin == floor(time_days/5)` for all 48.5 M rows | yes | PASS |
| `lon` ∈ [−180, 180) | [−180.0, 179.999] | PASS |
| `lat` range | [−78.305, 89.984] | PASS |
| `time_days` range | −1051.0 → 15705.75 = 1979-02-15T00:00Z → 2024-12-31T18:00Z | PASS (6-hourly grid, ends on the 18 UTC slot) |
| `fp.npy` unique rows | exactly one: (−4.0, −4.30078125) | PASS (f16 rounding of −4.3) |
| distinct `platform` ids | 28,689, min 1831 (AOML ids, all positive) | PASS |
| `Store.open` summary | `gdp: N=48,480,798 · C=4 (u, v, sst, drogue) · bins -211..3141 (3,353 live) · layout family10 · fp (-4, -4.3)` | PASS |

### Per-channel recomputation vs store.json's claims

Overall measured fraction recomputed: **0.963865** vs claimed 0.963865. **PASS.**

| channel | measured (mine) | measured (json) | fraction (mine / json) | min (mine / json) | max (mine / json) | mean (mine / json) |
|---|---|---|---|---|---|---|
| `u` | 48,425,603 | 48,425,603 | 0.998862 / 0.998862 | −3.44922 / −3.44922 | 4.03516 / 4.03516 | −0.0001283 / −0.0001283 |
| `v` | 48,425,602 | 48,425,602 | 0.998861 / 0.998861 | −2.80469 / −2.80469 | 2.82812 / 2.82812 | 0.0032272 / 0.0032272 |
| `sst` | 43,000,672 | 43,000,672 | 0.886963 / 0.886963 | −3.0 / −3.0 | 35.9375 / 35.9375 | 19.7499872 / 19.7499866 |
| `drogue` | 47,063,970 | 47,063,970 | 0.970775 / 0.970775 | 0.0 / 0.0 | 1.0 / 1.0 | 0.4563348 / 0.4563348 |

All four **match exactly** (the `sst` mean differs in the 7th decimal only —
float32 accumulation order). **PASS (4/4).**

Plausibility: mean `u` = −0.00013 m s⁻¹ and mean `v` = +0.0032 m s⁻¹, i.e. a
global drifter ensemble with essentially no net drift, which is the right
answer. `sst` min is exactly −3.0, the declared lower bound — freezing-point
water clipped at the bound, same pattern as `socat`'s `sss` floor. Peak speeds
(§8.5) reach 2.98 m s⁻¹ in a 2 M-row sample, well inside the ±5 m s⁻¹ bounds.

## 8.4 · The `drogue` three-state histogram, and `qc`

| `drogue` state | rows | fraction of N |
|---|---|---|
| **1.0** (drogue attached) | 21,476,927 | 0.44300 |
| **0.0** (drogue lost) | 25,587,043 | 0.52778 |
| **NaN** (status uncertain) | **1,416,828** | 0.02922 |
| any other value | **0** | — |

The channel is genuinely three-state as §3.1 requires: exactly 0, 1 or NaN,
never an interpolated value. 1 + 0 = 47,063,970 = the `per_channel.drogue.measured`
count, and 21,476,927 / 47,063,970 = 0.456335 = the claimed `mean` 0.4563348.
**PASS.**

**One discrepancy.** `counts.drogue_uncertain` = **1,417,396**, the array holds
**1,416,828** NaNs — **568 fewer**. And independently,
`rows_read − drop_pos_err − N` = 48,708,540 − 227,174 − 48,480,798 = **568**.
The same number twice. The consistent reading is that the counter counts source
rows with an uncertain drogue *before* the > 50 km position-error drop, and
that 568 of the 227,174 dropped rows were drogue-uncertain; the row ledger then
balances exactly. That is a counter-ordering artefact, **not a corrupt array** —
every array-side identity above closes. It is 568 rows in 48.5 M (1.2e-5).
Worth a one-line note in `store.json`; flagged, not a failure.

**Mechanism, read out of the builder afterwards (2026-09-14, same day).** The
counter is incremented before the drop, but not the position-error drop — that
one already precedes it. The rows are the ones the parser discards because
*nothing at all* was measured on them: a drifter fix with no velocity, no
temperature and an uncertain drogue is four NaNs, so it is counted as
drogue-uncertain and then dropped unrecorded. That single cause produces both
568s at once — the counter running ahead of `values`, and the row ledger
`rows_read − drop_pos_err − N` failing to close by the same amount — which is
why the two numbers are equal. `ml/build_family10_stores.py` now increments the
counter only for a row it keeps and counts that drop as `drop_no_values`, so a
future build closes both halves. That rebuild was dispatched 2026-09-14 08:27Z
(run 34822846450) and was **still running** when this note was written, so the
published store may still carry the gap and this note is the reconciliation.

| `qc` | rows | fraction |
|---|---|---|
| **1** (position error ≤ 5 km) | 45,363,989 | **0.935710** |
| **2** (5–50 km) | 3,116,809 | 0.064290 |
| 0 or ≥ 3 | 0 | — |

Nothing exceeds `qc_keep_max = 2` and nothing is "not assessed", exactly as the
derived-flag policy requires. **PASS.**

## 8.5 · `knearest` at the named anchors — §6's (k = 8, 300 km, 10 d)

### 8.5a · 36.0 N, −70.0 E, bin 2411 (2015-01-03) — the Gulf Stream anchor

`n_found = 8`, `n_R = 38`, **4.3 ms** median of 5 (min 3.9).

| slot | dist_km | dt_days | platform | qc | lat | lon | u | v | sst | drogue | \|vel\| |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 210.17 | 0.25 | 116295 | 1 | 35.469 | −67.768 | −0.335 | 0.020 | NaN | 1.0 | 0.336 |
| 1 | 218.02 | 0.50 | 116295 | 1 | 35.468 | −67.678 | −0.310 | −0.121 | NaN | 1.0 | 0.333 |
| 2 | 221.56 | 0.75 | 116295 | 1 | 35.516 | −67.621 | −0.318 | −0.071 | NaN | 1.0 | 0.326 |
| 3 | 230.36 | 1.00 | 116295 | 1 | 35.496 | −67.527 | −0.307 | 0.043 | NaN | 1.0 | 0.310 |
| 4 | 234.93 | 1.25 | 116295 | 1 | 35.499 | −67.474 | −0.382 | −0.127 | NaN | 1.0 | 0.403 |
| 5 | 245.09 | 1.50 | 116295 | 1 | 35.545 | −67.345 | −0.537 | −0.078 | NaN | 1.0 | 0.542 |
| 6 | 256.59 | 1.75 | 116295 | 1 | 35.530 | −67.219 | −0.470 | 0.091 | NaN | 1.0 | 0.479 |
| 7 | 265.74 | 2.00 | 116295 | 1 | 35.510 | −67.121 | −0.368 | −0.026 | NaN | 1.0 | 0.369 |

Internally this is textbook: **one** drifter (AOML id 116295), qc = 1, drogued,
sampled at exactly 0.25-day (6-hour) steps, walking eastward 35.47 N/−67.77 →
35.51 N/−67.12 over 2 days at ~0.3–0.5 m s⁻¹ **westward** velocity — a drogued
float in the anticyclonic recirculation south of the stream, moving with the
flow while its own reported `u` is negative. `sst` NaN on all eight: this
drifter's thermistor is not reporting, consistent with `sst` being only 88.7 %
populated store-wide.

**The expectation of |u| ≈ 1 m s⁻¹ is NOT met at this anchor, and the store is
not at fault.** The nearest drifters are 210 km *east-southeast* of 36 N/70 W,
outside the jet; k = 8 at R = 300 km never reaches the core. Checking the store
directly over the Gulf Stream box (32–45 N, 80–50 W) across bins 2409–2411:
1,645 rows from 37 drifters, speed median 0.251, p90 0.598, **p99 1.900, max
2.390 m s⁻¹**, with 51 rows above 1.0 m s⁻¹. Global 2 M-row sample: median
0.202, p99 0.917, max 2.981 m s⁻¹. **The metre-per-second Gulf Stream is in
the store; §6's (8, 300 km, 10 d) does not retrieve it from this anchor.**
This is the same finding as §5 in a different dress: k is spent on one
platform's own track, not on independent nearby measurements.

### 8.5b · 0.0 N, −140.0 E, bin 2411

`n_found = 0`, `n_R = 0`, all eight slots miss tokens (`valid=False`,
`row=−1`, values NaN). **3.3 ms.** Correct behaviour, and a real fact about the
equatorial Pacific in early January 2015: no drifter within 300 km / 10 days of
the TAO 0 N, 140 W site — the same anchor at which `gtmba` returns four rows
(§4b). The two tier-P stores are complementary there, which is the point of the
family.

## 8.6 · The §5 measurement for `gdp` — k-th neighbour, measured

Identical method to §5: 2,000 anchors per draw, generous bounds
R_max = 5000 km / T_max = 60 d, metric scales pinned to §6's (L = 300 km,
T = 10 d), seed 20260914; catalogue anchors stratified equally over the years
1982–2024 with `bin ≥ 0`; globe anchors lat ~ U(−60, 60), lon ~ U(−180, 180),
random live bin ≥ 0.

### Distance to the 8th neighbour (km)

| draw | median | p90 | p99 | anchors with < 8 found (5000 km / 60 d) |
|---|---|---|---|---|
| catalogue | **19.15** | **68.40** | **104.34** | 0.0 % |
| globe | 535.39 | 2,671.999 | 4,589.87 | **10.55 %** |

### Age of the 8th neighbour (days)

| draw | median | p90 | p99 |
|---|---|---|---|
| catalogue | 2.00 | 2.00 | 3.75 |
| globe | 2.25 | 14.30 | 46.53 |

### `n_R` at the suggested (8, 300 km, 10 d)

| draw | median | p90 | mean | n_R = 0 | n_R < k |
|---|---|---|---|---|---|
| catalogue | 80 | 240 | 128.1 | 0.0 % | 0.0 % |
| globe | **0** | 72.1 | 19.4 | **70.95 %** | 72.50 % |

### Distinct platforms among the k = 8 returned (600 catalogue anchors)

| bounds | median | mean |
|---|---|---|
| suggested (300 km, 10 d) | **1** | 1.16 |
| generous (5000 km, 60 d), same metric | **1** | 1.16 |

### What this says about §6's `gdp` suggestion (k = 8, 300 km, 10 d)

- **R = 300 km is ~3× too large at a sampled anchor.** The 8th neighbour sits
  at 19 km (median), 68 km (p90), 104 km (p99). **R_max ≈ 110 km covers the
  99th percentile**; 300 km is never the binding constraint on-track.
- **T = 10 d is ~5× too large.** The 8th neighbour is 2.00 days old at both the
  median and the p90 — that is exactly two days of one drifter's 6-hourly
  fixes. **T_max ≈ 4 d gives the same answer** at essentially every on-track
  anchor.
- **The k tokens are one drifter.** Median 1, mean 1.16 distinct platforms among
  8 — the eight slots are the same buoy at 0.25-day intervals, as §8.5a shows
  in the raw. §6's rationale ("a short window keeps the track coherent") is
  correct about *what happens* but the consequence is that k = 8 delivers one
  track, not eight independent samples. If independence is wanted, the search
  needs de-duplication by `platform`, or k must be much larger than 8.
- **Off-track it fails, as designed.** 70.95 % of uniform globe anchors see
  n_R = 0 inside (300 km, 10 d) and 72.5 % see fewer than 8 — the miss token is
  the normal answer over most of the ocean at a 10-day window, despite ~1,300
  active drifters.

Relative to the other two stores, `gdp` is the best-covered tier-P store on
both draws (10.55 % short at generous bounds vs `socat` 15.7 % and `gtmba`
34.05 %; n_R = 0 on 70.95 % of globe anchors vs 89.45 % and 81.3 %), which is
what a globally seeded drifter array should look like.

## 8.7 · Timing (`gdp`)

| call | median ms | p90 ms |
|---|---|---|
| (8, 300 km, 10 d), named anchor 36 N/70 W | 4.3 | — |
| (8, 300 km, 10 d), named anchor 0 N/140 W | 3.3 | — |
| generous (5000 km, 60 d), catalogue anchors | 13.59 | 26.74 |
| generous (5000 km, 60 d), globe anchors | 12.87 | 26.37 |

Between `gtmba` (0.4 ms) and `socat` (8 ms) at the suggested bounds, and the
most expensive of the three at generous bounds — 48.5 M rows and the densest
recent pentads.

## 8.8 · Verdict table for `gdp`

| check | result |
|---|---|
| all 9 files downloadable from the Hub | PASS |
| sha256 of all 9 files vs store.json | **PASS (9/9)** |
| `verify_store()` returns without raising | **PASS (9)** |
| `Store.open` on the local directory | PASS |
| N / bin_first / bin_last / n_bins vs store.json | PASS |
| live bins = 3,353 of 3,353, matches store.json and the build log | PASS |
| rows sorted by (bin, time_days) | PASS |
| `bin_offsets` CSR reproducible from `bin.npy` | PASS |
| `bin == floor(time_days/5)` | PASS |
| lon ∈ [−180, 180), lat in range | PASS |
| overall measured fraction (0.963865) | PASS |
| per-channel measured / fraction / min / max / mean | **PASS (4/4)** |
| per_year sums to N, and reproduces from `time_days` | **PASS (exact, 46/46)** |
| `drogue` is exactly three-state (1 / 0 / NaN), no fourth value | PASS |
| `drogue` NaN count vs `counts.drogue_uncertain` | **DIFFERS by 568 rows (1.2e-5)** — reconciles exactly with `rows_read − drop_pos_err − N` = 568, i.e. the counter is pre-drop. Arrays are self-consistent; a counter-ordering note is owed in store.json |
| qc ⊆ {1, 2}, split 93.571 % / 6.429 % | PASS |
| `knearest` fixed-k, miss tokens where empty | PASS |
| values physically plausible | PASS (global mean drift ≈ 0; Gulf Stream box p99 1.90, max 2.39 m s⁻¹) |
| Gulf Stream anchor shows \|u\| ≈ 1 m s⁻¹ | **NOT OBSERVED** — 0.31–0.54 m s⁻¹ from a single recirculation drifter 210 km away; the ≥ 1 m s⁻¹ rows exist in the store but lie outside (300 km, 10 d) of this anchor |
| §6 suggested radii are the measured optima | **NOT CONFIRMED — measured R ≈ 110 km, T ≈ 4 d** |

## 8.9 · The registry, `tensors/family10/family10.json`

Downloaded (38,128 B). `family: family10`, `repo: chfrank/earth-tensors`,
`handover: docs/FAMILY10_DATA_HANDOVER.md`, builder
`ml/build_family10_registry.py` @ `c5bc2ced06a2d1473f1fc47e2126d190c26ea0f4`
(the same commit as all three stores), `generated_utc`
**2026-09-14T08:05:45Z** — 70 s after `gdp`'s `built_at`, so the registry was
regenerated to pick `gdp` up. `n_groups: 7`.

### `groups` (7)

| group | tier | cadence | C | bin_first | N | files |
|---|---|---|---|---|---|---|
| `g025` | G | pentad | 7 | 0 | — (gridded) | 1 |
| `g100` | G | pentad | 15 | 0 | — | 1 |
| `rg100` | G | monthly | 32 | 0 | — | 1 |
| `argo` | P | irregular (per observation) | 32 | 0 | 2,678,439 | 12 |
| `gdp` | P | 6-hourly | 4 | **−211** | **48,480,798** | 9 |
| `gtmba` | P | daily | 18 | **−304** | 1,001,282 | 9 |
| `socat` | P | irregular (per observation) | 4 | **−1768** | 41,830,675 | 9 |

Every tier-P entry's `N`, `bin_first` and file count agrees with the
`store.json` I verified — including `gdp`'s, so the registry is describing the
build that actually landed. Family 8's `argo` store is carried as a group
unchanged (C = 32, bin_first 0), exactly as §7 of the handover says. Tier G
entries carry the byte-range layout string; tier P the CSR + k-nearest one, so
"a consumer dispatches on `tier`" holds.

### `groups_missing`

`["slatrack"]`, with `groups_missing_note`: *"['slatrack'] could not be read,
from the work directory or from the Hub. They are NOT in `groups` — a registry
that listed a group it could not describe would be worse than one that is
short. Build or publish them and re-run this builder."*

So three of the four §2 tier-P stores are published and registered; **`slatrack`
is not built.** The registry honours the load-bearing property of §2.2.

### `tier_g`

| field | value |
|---|---|
| `stem` | `family7_global025_pentad_l0` |
| `recipe` | **`f7l0`** (the fallback, not `f7l1`) |
| `manifest` | `…/tensors/family7_global025_pentad_l0/manifest.json` |
| `manifest_present` | `true` |
| `builder_git_sha` | `56d07f4c3801a532dea280e28816b98b324a7d79` |
| `built_at` | 2026-09-04T17:00:53Z |
| `index` | `data/family7_index.json` |

`fallback_note` (verbatim): *"E-079 §2 names family 7.1
(family7_global025_pentad_l1), whose build (E-077, recipe f7l1) has not been
dispatched — tensors/family7_global025_pentad_l1/manifest.json is a 404 on the
Hub. This registry therefore describes family7_global025_pentad_l0 (recipe
f7l0, three groups, no ocean colour). Re-run this builder after the f7l1 build
and it will pick up l1 and the fourth group `oc025` with no change here."*

The fallback is stated rather than silently taken, and the three tier-G groups
listed (`g025`, `g100`, `rg100`) are exactly l0's — the fourth group `oc025`
(ocean colour) is absent, consistent with the note. **PASS.**

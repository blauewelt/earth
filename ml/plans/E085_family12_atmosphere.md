# E-085 · Family 1.2 — family 1.gf plus the upper air (ERA5 temperature, humidity and wind on 13 pressure levels)

*Written 2026-10-05, after the adapters, the tests and a real one-month probe,
and BEFORE the full build. Chris asked for a new family: family 1.gf
unchanged, plus three more three-dimensional, six-hourly atmospheric
channels. This plan records the decisions the planning session made, what the
source archive actually holds (measured, not transcribed), what one real
month cost and weighed, and the build ladder for the rest. Nothing has been
dispatched, rented or uploaded.*

## 0 · In plain English

- **Family 1.gf** is the set of global *observation* stores at 10 km or finer
  and five days or finer — ocean colour, satellite sea-surface temperature,
  cloud-top temperature, ship reports, floats, bottles, moorings and the
  rest — each kept at its own resolution and published on the Hugging Face
  dataset `chfrank/earth-tensors` under `tensors/family1_gf/`.
- **Family 1.2** is family 1.gf, **inherited by reference** (no byte copied:
  its registry points at 1.gf's), plus four new stores holding the state of
  the atmosphere above the surface, which no instrument observes globally
  every six hours.
- **ERA5** is ECMWF's reanalysis: a weather model run over the whole
  observational record and pulled towards every observation the centre holds,
  so it gives a complete, physically consistent atmosphere for every hour
  since 1940. It is licensed CC BY 4.0 by the EU's Copernicus Climate Change
  Service. Google publishes an analysis-ready copy, **ARCO-ERA5**, as zarr
  archives (chunked compressed arrays) on a public Cloud Storage bucket that
  anyone can read over HTTPS without an account.
- **The channels.** Air temperature `t`, specific humidity `q` (grams of
  water vapour per kilogram of air) and the wind as its eastward `u` and
  northward `v` components, each on the **13 standard pressure levels** 50,
  100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925 and 1000 hPa — from
  about 20 km up (50 hPa) to the surface (1000 hPa). `t_500` is temperature
  at 500 hPa.
- **The frames.** One frame per analysis instant at 00, 06, 12 and 18 UTC:
  20 per five-day bin (the time unit every family files under), from
  1982-01-01 (the families' epoch) to the archive's last final hour,
  2026-06-30 18 UTC.
- **The grid.** One degree, 181 × 360, point-aligned like family 7's `g100`
  (the 1° group of the global tensor): latitude −90…90, longitude −180…179.
  Each value is the **conservative** (area-weighted) mean of ERA5's
  0.25° field over the 1° box around the point.
- **The layout** is the existing sharded tier-G layout
  (`ml/family1/sharded.py`): one compressed shard per five-day bin, 64 × 64
  tiles, float16 in physical units. One store per variable; the 13 levels are
  the 13 channels.
- **Exception E4.** 1.gf's rule admits observations at ≤ 10 km. ERA5 is a
  model-filled reanalysis at ~31 km, stored as ~111 km box means. It is
  admitted **by name** as exception E4 (the 1.gf note already names E1–E3:
  12.5 km scatterometer wind, IMERG precipitation, GOSAT CO₂).

Links:

- [the adapter's shared module](https://github.com/blauewelt/earth/blob/main/ml/family1/adapters/_era5.py)
- [the extra probe measurements](https://github.com/blauewelt/earth/blob/main/ml/family1/era5_check.py)
- [the tests](https://github.com/blauewelt/earth/blob/main/tests/test_family12_era5.py)
- [the January-2015 probe of `era5_t`](https://github.com/blauewelt/earth/blob/main/ml/family1/probes/era5_t_2015-01.json) (and `era5_q`, `era5_u`, `era5_v` beside it)

## 1 · Decisions

**Numbered decisions of the planning session, 2026-10-05, on the probe report
(§4).** These are the ones a later session must not quietly reopen:

- **D1 — build the four stores exactly as probed:** 1°, six-hourly instants,
  the 13 levels, one store per variable, float16, humidity in g/kg.
- **D2 — the shard byte order does NOT change.** The planar, byte-shuffled
  tile order of §4 would save ~26 % of 349 GB, which is not worth a format
  fork across every reader of `sharded.py`. It stays a recorded option.
- **D3 — geopotential stays out** (it does not fit float16 without a
  transform of its own).
- **D4 — ERA5T stays out:** the record ends where final ERA5 ends
  (`valid_time_stop`, 2026-06-30).
- **D5 — specific humidity's lower bound is −0.01 g/kg, not 0.** ERA5's own
  small negative humidities (January 2015: 11 values, the lowest −0.0042
  g/kg, at 100–200 hPa in the tropics) are kept as ERA5 gives them;
  anything below −0.01 g/kg is corrupt (NaN, counted).

The table below is the full set of choices, with what each rests on;
the measurements in §2 and §4 are its evidence.

| decision | choice | why |
|---|---|---|
| source | ARCO-ERA5 on `gcp-public-data-arco-era5`, anonymous HTTPS | CC BY 4.0, keyless, chunked for ranged reads; verified reachable from the sandbox |
| channels | `t`, `q`, `u`, `v` on 13 levels = 52 level-channels | the decision; wind is one channel of two components |
| geopotential | **not added** | measured: 52 % of its values exceed float16's maximum (65,504) in m²/s² (50 hPa reaches 207,221), and as a height in metres the float16 step is 16 m at 50 hPa and 4 m at 500 hPa — it would need its own transform, which is extra design, so the rule ("only at no extra design cost") leaves it out |
| frames | the analysis INSTANT at 00/06/12/18 UTC; F = 20, `frame_seconds` 21,600 | the decision; frame f of bin b = b·432,000 + f·21,600 s since 1982-01-01 |
| record | 1982-01-01 00 UTC → 2026-06-30 18 UTC, 65,008 frames | the epoch, to the 0.25° archive's `valid_time_stop` (final ERA5) |
| ERA5T | **not admitted** (frames after 2026-06-30 are `after_record`) | ERA5T is ECMWF's preliminary release (here to 2026-09-29) and may be revised; admitting it is one setting to change later |
| sources and seam | before 2022-01-01 00 UTC: archive **A** (the producer's own 1° conservative regrid, hourly); from it: archive **B** (0.25°, hourly) regridded here | A is exactly the store's grid and costs ~5–6 MB a frame; it ends 2021-12-31 23 UTC. The seam is READ from A's time axis, and falls exactly on the start of bin 2922 |
| grid | 1°, 181 × 360, row 0 = −90, column 0 = −180 | point-aligned like `g100`; 0.25° is phase B (§6) |
| regrid rule (for B) | separable conservative weights from the archives' own axes: latitude edges at midpoints clipped to ±90, overlap in sin(latitude); longitude edges at midpoints, periodic, overlap in degrees; each row normalised; applied as fixed-order elementwise sums (no matrix library, so the bytes do not depend on the CPU); a NaN in any contributing cell makes the cell NaN | "what did this cell average" is answered by an area mean; measured equal to A's own regrid to 1.4 × 10⁻³ K and 3 × 10⁻³ m/s (§4) |
| layout | **one store per variable**, C = 13, sharded tier G, `tile` 64 | a reader that wants one variable downloads one; the contract makes nothing worse. 64-pixel tiles are 64° on a side — the same footprint in degrees as a 256-pixel tile of the 0.25° phase B — so a cone (the region the model reads around a forecast point) of a few thousand km reads 1–4 tiles; measured 4.6 % SMALLER than 256-pixel tiles too (§4) |
| dtype | float16 in physical units: K, g/kg, m/s | the decision; measured error is half a float16 step everywhere (§4) |
| humidity transform | **store q in g/kg** (archive kg/kg × 1000) | float16's smallest normal number is 6.1 × 10⁻⁵; ERA5's q runs from ~2 × 10⁻⁶ kg/kg (stratosphere) to 0.024 kg/kg. In kg/kg every stratospheric value is a subnormal with ≥ 0.5 % error (test: up to 1.5 %); in g/kg every value from 10⁻⁷ kg/kg up is normal and the relative error is ≤ 2⁻¹¹ = 0.049 % (measured: 4.88 × 10⁻⁴). A unit change, not a nonlinear transform, so a reader needs one division to get kg/kg |
| q's lower bound | −0.01 g/kg, not 0 | ERA5 itself carries a few slightly negative humidities (January 2015: 11 values in 10.1 M, the lowest −0.0042 g/kg, all at 100–200 hPa in the tropics). They are ERA5's numbers, so they are kept; below −0.01 g/kg is treated as corrupt (NaN, counted). Reversible: one constant |
| other bounds | t 150–350 K; u, v ±200 m/s | physical envelopes; outside → NaN and counted, never clipped (contract rule 3) |
| below-ground levels | kept as ERA5 gives them | ERA5 extrapolates pressure levels under the terrain (1000 hPa over Tibet); the store keeps them and says so; a consumer masks with surface pressure |
| footprint | `log2_fp` 2.0 (a 111 km box); `log2_dt` −4.32 (six hours) | the cell IS a 1° box mean; an instantaneous analysis has no averaging time of its own, so the time field is the frame spacing (the store.json definition, "log2(frame_days / 5)"); the registry therefore reads "6-hourly" |
| family | code `12`, version `1.2`, root `tensors/family1_2/`, registry `tensors/family1_2/family12.json` | the decision |
| licence | `redistribution: attribution`, `derived_works: free`, `distribution: public`, with the Copernicus attribution string and the Hersbach et al. (2020) citation | CC BY 4.0 |

## 2 · What the archive holds — measured 2026-10-05

Read from each zarr's own consolidated metadata (`.zmetadata`) and its
decoded axes, from this sandbox, anonymously.

**The bucket's `ar/` prefix holds 19 zarr groups.** The ones that matter:

| zarr | cadence | grid | levels | dims and chunk | compressor | time axis |
|---|---|---|---|---|---|---|
| **A** `1959-2022-1h-360x181_equiangular_with_poles_conservative.zarr` | hourly | 1°, lat −90…90 ascending (181), lon 0…359 (360) — conservative, with poles | 37 | `(time, level, longitude, latitude)` float32, chunk [8 h, 37, 360, 181] = 77.1 MB raw | blosc lz4, byte shuffle, 512 KiB blocks | hours since 1959-01-01; 552,264 steps, contiguous; **1959-01-01 00 → 2021-12-31 23** |
| **B** `full_37-1h-0p25deg-chunk-1.zarr-v3` | hourly | 0.25°, lat 90…−90 **descending** (721), lon 0…359.75 (1440) | 37 | `(time, level, latitude, longitude)` float32, chunk [1 h, 37, 721, 1440] = 153.7 MB raw | blosc lz4, byte shuffle, 512 KiB blocks | hours since 1900-01-01, PRE-ALLOCATED 1900 → 2050 (1,323,648 steps); root attributes: `valid_time_start` 1940-01-01, **`valid_time_stop` 2026-06-30**, `valid_time_stop_era5t` 2026-09-29, `last_updated` 2026-10-05 11:58 |
| `1959-2022-6h-1440x721.zarr` and `1959-2022-wb13-6h-0p25deg-chunk-1.zarr-v2` | six-hourly | 0.25°, lat descending | **13** (exactly ours) | `(time, level, latitude, longitude)`, chunk [1, 13, 721, 1440] | blosc lz4 | 1959-01-01 00 → 2021-12-31 18 (92,044 steps); identical chunk sizes in both |
| `1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr` | six-hourly | 1.5° | 13 | `(time, level, longitude, latitude)`, chunk [8, 13, 240, 121] | blosc lz4 | 1959-01-02 00 → 2021-12-31 18 |

The 37 levels of A and B are 1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125,
150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750,
775, 800, 825, 850, 875, 900, 925, 950, 975, 1000 hPa; all 13 of ours are
among them and are picked BY VALUE. Every variable we need — and
`geopotential` — is in both.

**Why A and not the six-hourly 13-level 0.25° archive.** No source holds
exactly 1° at six hours. A holds exactly 1° hourly, and an hourly archive
holds every six-hourly instant. Compressed sizes of one frame's worth, all
four variables, 2015-01-01 00 UTC: the six-hourly 0.25° archive's chunks are
29.8 (t) + 38.4 (q) + 41.8 (u) + 42.7 (v) = 152.7 MB, to be read whole and
regridded; A's 8-hour chunks are 48.3 / 59.1 / 64.7 / 65.8 MB, but the
block reader (§3) fetches only the blocks holding the wanted hour and
levels: **4.7–6.5 MB per variable per frame, measured**. Reading A costs
~8 % of the alternative.

**The seam.** A ends 2021-12-31 23 UTC; its next hour, **2022-01-01 00 UTC,
is exactly the start of bin 2922** (14,610 days after the epoch), so no
five-day bin mixes the two sources in the real store. The synthetic smoke
moves its seam inside a bin to test the harder case.

## 3 · How a frame is read — only the bytes the store needs

A blosc chunk is a 16-byte header, a table of block start offsets, and
independently compressed blocks of 512 KiB of the uncompressed C-order
buffer. The reader:

1. reads the chunk's first 4 KiB (the header and the whole table: 148
   blocks for A, 294 for B) and checks it against the zarr's declaration —
   typesize 4, uncompressed size = the chunk shape, compressed size = the
   object's size from the response's `Content-Range`; a table that does not
   tile the chunk exactly is a refusal;
2. works out which blocks hold the wanted (hour, level) slabs (13 of A's
   8 × 37, or 13 of B's 37), and fetches them with ranged GETs, byte-adjacent
   blocks coalesced — **found while testing:** a multi-threaded blosc writer
   stores blocks out of index order, so a block's end is the next-higher
   START, not the next index's;
3. decodes each run of consecutive blocks by re-wrapping it as a small valid
   blosc chunk handed to `numcodecs` (the only decoder used, so decoding is
   c-blosc's own). Measured on real chunks: bit-identical to decoding the
   whole chunk.

Every read must answer 206 with exactly the bytes asked for; a 200, a short
body, a 404 inside the valid record or an undecodable run is a REFUSAL — the
frame's input is noted absent, the year is not marked, the fetch stage exits
— never a skip. Bytes are counted per call (§4 records it), and 16 threads
read frames in parallel while the main thread encodes.

## 4 · The probe — January 2015, real archive, real adapter, real writer

Run from the sandbox (2 vCPU, 7 GB RAM) on 2026-10-05:
`build_family1_stores.py --store era5_<v> --stage probe --probe-month
2015-01` (124 frames through the real writer), then `--stage
index,fetch,assemble --start 2015-01-01 --end 2015-01-31` into a local work
directory (lane `m01`: the six bins starting in January, 120 frames), then
`ml/family1/era5_check.py` on that store. Everything below is in
`ml/family1/probes/era5_<v>_2015-01.json`. Two caveats on provenance: the
probes ran from the uncommitted working tree on top of 4a3be69 (the
`builder_git_sha` they record), and the sandbox has zstandard 0.25.0 where
the workflow pins 0.23.0 — ADAPTER_CONTRACT.md measured that the two
compress about 10 % of tiles to different bytes, so the sizes carry over and
the exact shard bytes do not.

| | `era5_t` | `era5_q` | `era5_u` | `era5_v` |
|---|---|---|---|---|
| frames written / missing | 124 / 0 | 124 / 0 | 124 / 0 | 124 / 0 |
| out of bounds (NaN, counted) | 0 | 0 | 0 | 0 |
| valid fraction | 1.0 | 1.0 | 1.0 | 1.0 |
| stored bytes per frame | 0.832 MB | 1.446 MB | 1.525 MB | 1.559 MB |
| stored bytes per tile (mean of 18) | 46.2 kB | 80.3 kB | 84.7 kB | 86.6 kB |
| compression vs raw 1.694 MB | 2.04× | 1.17× | 1.11× | 1.09× |
| **the month** (124 frames + indices) | **103 MB** | **179 MB** | **189 MB** | **193 MB** |
| **full record, extrapolated** (65,008 frames) | **54.1 GB** | **94.0 GB** | **99.2 GB** | **101.3 GB** |
| source bytes per frame, path A (1°) | 4.73 MB | 5.97 MB | 6.41 MB | 6.51 MB |
| source bytes per frame, path B (0.25°, 24 frames of 2023-01) | 32.3 MB | 41.2 MB | 45.4 MB | 46.7 MB |
| source bytes, full record | 0.49 TB | 0.62 TB | 0.67 TB | 0.69 TB |
| probe wall time for 124 frames | 35.2 s | 35.5 s | 27.3 s | 28.4 s |
| float16 error, max abs | 0.125 K | 0.0078 g/kg | 0.031 m/s | 0.031 m/s |
| float16 error, max relative | 4.88 × 10⁻⁴ | 4.88 × 10⁻⁴ | 4.88 × 10⁻⁴ | 4.88 × 10⁻⁴ |
| read-back vs independent read, max abs diff | 0.125 K | 0.0078 g/kg | 0.031 m/s | 0.031 m/s |
| … every value within half a float16 step | yes | yes | yes | yes |
| seam: path A vs path B, max abs diff (2015-01-01 00 / 2021-12-31 18) | 1.4e-3 / 3.8e-4 K | 1.1e-4 / 1.3e-4 g/kg | 2.6e-3 / 3.0e-3 m/s | 1.7e-3 / 2.5e-3 m/s |
| … values identical after float16 rounding | 99.99 % | 99.86 % | 99.05 % | 99.02 % |
| peak RSS, probe / fetch+assemble | 688 / 783 MB | 702 / 799 MB | 682 / 764 MB | 676 / 806 MB |

**Read-back.** One frame (2015-01-15 12 UTC, bin 2413 frame 10) through
`sharded.ShardedGroup.read_frame`, against the same instant read
**independently** with xarray 2026.9.0 + zarr 3.4.0 from archive A (levels
selected by value, longitudes re-labelled and sorted, no adapter code): the
largest difference per variable is exactly the float16 rounding — half a
step at the value — over all 847,080 values.

**Sanity.** January 2015: `t_500`'s area-weighted global mean 258.1 K
(frames 257.9–258.4; unweighted ≈ 252 K); `t_1000` 286.8 K; `t_50` 210.5 K,
min 189.9 K. `u_200`'s maximum **106.5 m/s at 33° N 147° E** (the Japan jet),
minimum −45.3 m/s, global mean +16.6 m/s; `v_200` −84.4 to +77.9 m/s.
`q_1000` up to 23.6 g/kg (mean 8.9); `q_50` 0.0024–0.0032 g/kg; `q_150`'s
minimum −0.0042 g/kg (ERA5's own, kept, §1). One `q_100` value and 17–112
near-zero u/v values per level are float16 subnormals (|x| < 6.1 × 10⁻⁵),
absolute error ≤ 3 × 10⁻⁸ — counted in `f16_subnormal_values`, harmless.

**The seam is invisible at storage precision.** At two instants both archives
hold, this module's regrid of B against the producer's regrid A differs by at
most 1.4 × 10⁻³ K, 1.3 × 10⁻⁴ g/kg and 3.0 × 10⁻³ m/s — float32
accumulation noise in the producer's regrid (relative 6 × 10⁻⁶ for
temperature), below the float16 step except for the smallest winds — and
99.0–99.99 % of stored values are bit-identical whichever source is used.
The rule is reproduced, not approximated.

**Tiles.** The same eight frames: 64-pixel tiles 0.831 MB/frame (t), 256-pixel
tiles 0.871 MB/frame — 64 is both more local and smaller.

**THE COMPRESSION FINDING (a decision for Chris, not taken here).** Humidity
and wind compress only 1.09–1.17× because the layout stores a tile in C order
`[row, col, channel]`: the 13 levels' float16 values interleave byte by
byte, and zstd sees noise. Measured on the same tiles, what a layout option
would buy (bytes per frame, 64-pixel tiles):

| order | t | q | u | v | full record, all four |
|---|---|---|---|---|---|
| C order (what `sharded.py` writes) | 0.831 | 1.448 | 1.525 | 1.558 MB | **349 GB** |
| channel-planar `[channel, row, col]` | 0.597 | 1.258 | 1.501 | 1.549 MB | ≈ 321 GB |
| planar + byte shuffle (the two bytes of each float16 in two planes) | 0.538 | 1.007 | 1.167 | 1.248 MB | **≈ 257 GB (−26 %)** |

A per-group byte-order option in `tile_grid.json` would be a change to
`ml/family1/sharded.py` and to every reader of it (the Data tab's
`src/f1data.js` included) — out of this session's scope, and the 349 GB
figure is what the store costs as designed.

## 5 · Storage arithmetic

- **Frames.** 1982-01-01 00 → 2021-12-31 18 from A: 58,440; 2022-01-01 00 →
  2026-06-30 18 from B: 6,568; total **65,008** (16,252 days × 4).
- **Stored.** 0.832 / 1.446 / 1.525 / 1.559 MB per frame × 65,008 + 3,251
  bins × 5,888 index bytes → **54.1 + 94.0 + 99.2 + 101.3 = 348.6 GB** on the
  public Hub (inside the PRO 10 TB), 1.17 M stored tiles per store, about
  6,500 files per store (a shard and an index for each of 3,251 bins, 73
  bins per year folder — far under the Hub's 10,000-per-folder limit).
- **Per year of parts** (what a lane holds on disk before it pushes): 1.22
  (t), 2.11 (q), 2.23 (u), 2.28 (v) GB.
- **Read from the source.** 58,440 × 4.7–6.5 MB + 6,568 × 32–47 MB =
  **0.49 + 0.62 + 0.67 + 0.69 = 2.47 TB**, of which 1.08 TB is the 4.5-year
  tail from B (13 of 37 levels at 0.25°, 114 of 294 blocks a frame).
- **Phase B, 0.25°** (recorded, not built): the same frames on 721 × 1440
  are 16 × the pixels; with the measured ratios that is ≈ 0.86–1.6 TB per
  store, ≈ 5.6 TB for four, before any byte-order option — a decision about
  Hub space, not about code (the regrid step is simply skipped).

## 6 · The build ladder

Step 1 is running (§8); steps 2–4 have not been dispatched.

1. **Hosted fetch lanes** (`.github/workflows/family1-build.yml`, `runner:
   ubuntu-latest`, `stage: index,fetch`, `extra_args: --push-parts`, the
   store and a window of whole calendar years — an unnamed lane per window,
   parts parked at `partials/family1_2/<store>/<year>/`, every part
   restore-hashed and each year's `done.json` written last). The source is
   keyless, so the lanes need no secret.

   **The hosted runner, MEASURED by the canary #920 (era5_t 1982–1991, the
   first lane, 2026-10-05):** `/dev/root` 145 GB with **85 GB free** (91.0 GB
   decimal at the parts directory), **4 vCPU, 15 GiB RAM**. The "~14 GB"
   planning figure was wrong and the ADAPTER_CONTRACT's "~86 GB" is right.
   Rates: `era5_t` 212 s per year of fetch (0.145 s a frame, ~33 MB/s of
   source); `era5_q`/`era5_v` 117–133 s per year; the push of a lane's parts
   (upload, then download back and hash every file, then `done.json`)
   ~145 s per GB-year of t (1,450 s for the canary's 12.1 GB). The build
   step now prints `df -h`, `nproc` and `free -g` before the first byte, and
   the ERA5 preflight sizes every lane's parts against the free disk and
   refuses in seconds if they do not fit.

   **Re-planned from those numbers:** disk no longer binds (a 5-year lane
   is ≤ 11.4 GB of 85). Lanes are **5 calendar years** for all four stores —
   about 30–55 min each — chosen for parallelism under the 12-lane ceiling
   and to keep a failure's blast radius small, not for disk. 34 lanes after
   the canary (t 1992–2026 in 7, q/u/v 1982–2026 in 9 each; the last lane of
   each is 2022–2026, the 0.25° tail), queued on the `f1-queue` branch's
   `lane_queue.json` and kept at ≤ 12 in flight by the repository's
   queue keeper (`family1-queue.yml`, `max` 12). Each lane carries
   `retries: 2`, so the keeper re-dispatches a fast (< 20 min) failure once
   and then marks it failed; a slow failure is failed at once, and the log is
   read before any re-dispatch. Cost $0.

2. **Assembly on one rented, verified box, one store at a time** — prepared,
   NOT rented. Per store `S` in `era5_t`, `era5_q`, `era5_u`, `era5_v`, on the
   box's runner `gpu-box-<offerId>`:

   ```json
   {"store":"S","stage":"all","start":"1982-01-01","end":"2026-12-31",
    "runner":"gpu-box-<offerId>","extra_args":"--parts-from-hub --assemble streaming"}
   ```

   (`stage=all` on a tier-G store is index, fetch — which with
   `--parts-from-hub` PULLS the parked parts and reads no source —,
   assemble and publish, the publish uploading and downloading back every
   file; both `start` and `end` must be passed or the workflow's default end
   of 2024-12-31 silently drops 2025–2026), then the full decode by name:

   ```json
   {"store":"S","stage":"check","start":"1982-01-01","end":"2026-12-31",
    "runner":"gpu-box-<offerId>","extra_args":""}
   ```

   and, before the next store, `free_cache: "family1_2/S"` on that store's
   next dispatch (the step deletes a work directory only if it carries
   `publish.done` AND `check.done`).

   **The box:** a *verified* Vast host (ml/CLAUDE.md §7), no GPU needed
   (the work is pull, hard-link, hash, upload, download-back, decode);
   **disk ≥ 200 GB** — the largest store's parts are ~102 GB (`era5_v`,
   hard-linked into the store, not copied), plus the publish's one-year
   restore scratch (~2.3 GB), the checkout and pip environment (~10 GB), and
   ~2× headroom so a store can be checked while the next one's pull starts;
   or ≥ 450 GB to keep all four without `free_cache`; **RAM ≥ 16 GB** (the
   assembler links files and the check decodes 64 × 64 × 13 tiles one at a
   time — the January store peaked at 0.8 GB); **inet_down ≥ 1 Gbps**
   (each store is pulled once and downloaded back once by publish, ~200 GB
   per large store) and **inet_up ≥ 300 Mbps** (the upload is mostly
   de-duplicated by the Hub — the parts ARE the store files — so it is
   metadata plus whatever the Hub does not already hold). `node
   scripts/gpu_box.mjs offers` prints `inet_up`, `inet_down` and
   `verification`; its filter asks for an RTX 4090 and must be widened for a
   CPU-only job. Estimate **2–3 h per store, 8–12 h for four, ≈ $2–6**.

3. **The registry**, after all four stores are published and checked:
   `python3 ml/build_family1_registry.py --check` then `python3
   ml/build_family1_registry.py --publish`, which writes and
   restore-verifies `tensors/family1_2/family12.json` (and rewrites the
   other three family-1 registries, content unchanged apart from their
   timestamps). It needs `HF_TOKEN` with write access to
   `chfrank/earth-tensors`. This sandbox does not hold one, and no workflow
   runs the registry builder today (family1-build's steps hand `HF_TOKEN`
   only to `build_family1_stores.py`), so it needs either a machine holding
   the token or a small registry mode added to a workflow — an open item, not
   a blocker for the stores themselves. Offline
   today it lists 4 stores, `inherits: family1gf`, the E4 exception,
   `levels_hpa` per store and each probe.
4. **Phase B** (0.25°) only on Chris's word, and only after the byte-order
   decision in §4.

## 7 · Verification, and what would falsify the build

- **Tests** (all pass; §8): the frame/bin rule, the seam measured from the
  archive's own axis, levels selected by value from a 16-level synthetic
  archive, the regrid against a brute-force spherical-area implementation on
  a toy (rtol 10⁻¹²), NaN propagation, bitwise-stable regrid, the g/kg
  round-trip bound, the block reader against whole-chunk decoding on
  out-of-order blosc blocks, a 200 refused, a short range refused, a deleted
  and a truncated chunk each refusing the year, and the full smoke across a
  mid-bin seam with `after_record` past `valid_time_stop`.
- **Before a lane is trusted**: its ledger's `frames_from` must sum to 20 ×
  the bins whose first day is in its window, all from A before 2022 and all
  from B after; `inputs_not_read` empty; `out_of_bounds` per channel
  read (a non-zero `t` or `u`/`v` count is a corrupt chunk, not weather).
- **The assembled store is wrong if**: any year's frame count differs from
  4 × its days in its bins; `check` finds a tile out of bounds; a read-back of
  any frame against xarray on the source (as `era5_check.py readback` does)
  differs by more than half a float16 step; `t_500`'s January global mean
  leaves 255–261 K or `u_200`'s monthly maximum leaves 60–130 m/s in any
  year (a level or a sign mix-up would do either); or the frames either side
  of 2022-01-01 differ in global means by more than their neighbours do.

## 8 · Status

**2026-10-05, fetch phase running; nothing assembled, no box rented.**

- **Built and on main:** the adapters (`_era5.py`, `era5_t/q/u/v.py`), family
  code 12, family 1.2 in the registry builder (offline it lists 4 stores,
  `inherits: family1gf`, E4, `levels_hpa`), `era5_check.py`, the four
  January-2015 probe reports, the tests, `numcodecs` and an early `df -h` /
  `nproc` / `free -g` in the workflow, and a disk preflight that sizes each
  lane from the probe's bytes per frame (c8ffcd2).
- **Decisions D1–D5 (§1)** recorded on the probe report: build as probed,
  byte order unchanged, no geopotential, no ERA5T, q's lower bound
  −0.01 g/kg.
- **The canary #920** (era5_t 1982–1991) is parked and verified (its
  effect read off the Hub: ten `done.json`, 14,620 frames, 12.08 GB, one part
  re-hashed here), and it measured the hosted runner: 85 GB free, 4 vCPU,
  15 GiB. The other 34 five-year lanes are on the queue keeper at ≤ 12 in
  flight. The lane ledger, with each lane's run, status, frames and parked
  bytes, is the E-085 entry at the end of
  [the family-1 build log](https://blauewelt.github.io/earth/docs.html?f=ml/family1/BUILD_LOG.md).
- **The 0.25° tail path ran in real lanes** (2022 for all four stores,
  361–433 s a year) with stored sizes matching the 1° years'.
- **Next:** when every lane is parked (every year 1982–2026 of every store
  has a `done.json`), assemble per §6 step 2 on one verified box, one store
  at a time, then `--stage check`, then the registry (§6 step 3, which needs
  a machine holding `HF_TOKEN`).
- **Not measured:** nothing about the assembly yet (its time and cost are the
  §6 estimates).

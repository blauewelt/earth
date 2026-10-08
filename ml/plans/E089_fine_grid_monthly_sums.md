# E-089 · Long-period averages made cheap for the FINE grids: monthly sums of family 1.gf's four 2–4 km stores

Written 2026-10-08 (Fable plans, Opus implements — `ml/CLAUDE.md` §0b). The
build E-088 §8 recommended (`ml/plans/E088_monthly_sums_gridded.md`).

Chris, 2026-10-08: *"Please add the family 1.gf grids with subagents."*

## 1. In plain English

E-088 made a long-period average of eight 0.25°–1° stores cost a few range
reads instead of every native frame, by publishing, per year and calendar
month, the SUM of each cell's values, how many there were (the COUNT) and the
sum of squared deviations from that month's own mean (M2) — from which a mean
and a standard deviation over ANY set of months is composed exactly. It
measured, but did not build, the same thing for the four fine **family 1.gf**
grids (family 1.gf is the set of global observation stores finer than 0.25°,
each at its own resolution):

| store | what it is | native grid | channels | cadence | record (to the store's real end) |
|---|---|---|---|---|---|
| `oc4k` | ESA OC-CCI v6 ocean colour: log₁₀ chlorophyll-a, Kd(490), observation count | 1/24° (4 km), 4320 × 8640, north-first | 3 | daily | 1997-09-04 → 2022-12-31 |
| `pace4k` | NASA PACE OCI ocean colour and phytoplankton community | 1/24°, the same grid | 7 | daily | 2024-03-05 → 2026-07-31 |
| `sst_acspo02` | NOAA ACSPO L3S-LEO sea-surface temperature and its quality grade | 0.02° (2 km), 9000 × 18000, north-first | 2 | daily | 2000-02-26 → 2026-09-16 |
| `irtb` | NCEP/CPC merged geostationary infrared — cloud-top brightness temperature, the ±30° band | 360/9896° (≈ 4 km), 1649 × 9896, south-first | 1 | 3-hourly | 1998-01-02 → 2026-09-16 21 UTC |

A dense monthly plane of these is 37–162 million cells, mostly cloud, night
and land, so E-088's dense layout alone would be 754 GB. E-089 builds the two
layers E-088 §8 recommended:

1. **Native monthly tiles** — at each store's own resolution, one file per
   (year, month) of compressed tiles in the store's own 256 × 256 tiling,
   tiles with no observation not stored. A box at full resolution is a few
   tiles per year-month, read the way the Data tab already reads the stores.
2. **An exact 0.25° pooled layer** in E-088's dense layout, whose sums and
   counts are EXACTLY the native layer block-summed (never a mean of means),
   so a global long-period map costs what a 0.25° store costs.

## 2. Decisions

1. **The month is E-088's**: the true calendar month (UTC) of each frame's
   own start instant; every native frame of that month. All data to each
   store's real record end (the registry's `record_span`), not to 2024.
2. **Physical units.** float16 stores as stored. `irtb` stores uint8 = kelvin
   − 160; it is summed in **kelvin** (`value_offset` 160, unit `K`). Its sums
   are integers below 2²⁴ (414 K × 248 frames × 49 pixels), so even its
   pooled float32 sums are exact.
3. **Native counts are uint8 — asserted.** At most 31 daily frames, or 248
   three-hourly `irtb` frames in a 31-day month (7 to spare). The exporter
   refuses a month with more than 255 frames.
4. **sum f32 / count / m2 f32 exactly as E-088 §2.4–2.6**: float64
   accumulation of float16 values is exact; one float32 rounding; m2 is the
   two-pass float64 Σ(x − mean_ym)² about each (year, month)'s own mean;
   sum = m2 = 0 where count = 0, m2 = 0 where count = 1.
5. **The native layer is `sharded.py`'s layout with a calendar month in place
   of a five-day bin** (`ml/family1/monthly_tiles.py`, which reuses
   `sharded.py`'s tiling, index-header, source and codec code). The three
   statistics take the place of the F frames as PARTS, and the parts of one
   tile are ADJACENT in the file, so a mean (sum + count) is one range read
   per tile and a standard deviation the same read extended by m2.
6. **The pooling rule — one rule for all four grids.** A 0.25° cell holds
   EVERY native pixel whose CENTRE lies in it; cells are half-open
   [south, north) × [west, east) on the cell-registered 0.25° grid (edges at
   multiples of 0.25°, centres at −89.875°, …), and a centre within 10⁻⁶ of a
   cell (2.5 × 10⁻⁷°) of an edge is ON that edge, i.e. in the cell north / east
   of it. The geometry, checked on the four real `tile_grid.json`s:

   | store | native pixel | pixels per 0.25° | pooled grid | blocks |
   |---|---|---|---|---|
   | `oc4k`, `pace4k` | 1/24° | 6 | 720 × 1440, whole globe | **exact 6 × 6** — native edges fall on every 0.25° edge |
   | `sst_acspo02` | 0.02° | **12.5** | 720 × 1440 | 12 / 13 alternating (centre-binned): 0.02° edges coincide with 0.25° edges only every 0.5°, so a pixel is cut in half by every other cell edge and goes where its centre is — the brief's "12 × 12" would be 0.24° cells |
   | `irtb` | 360/9896° ≈ 0.0364° lon, 120/3298° lat | 6.87 | **240 × 1440: the ±30° band** (cell rows −30° … 30°, global cell rows 240–479), lat0 −29.875° | 6 / 7 alternating; the band's first row is centred exactly on −30° (an edge) and goes to the cell [−30°, −29.75°) |

   For the two centre-binned grids a pooled mean is the mean over the native
   SAMPLES whose centres are in the cell — an exact statistic of the native
   data, reproducible from the native layer by the published `rows` / `cols`
   (each pooled row's / column's native [start, stop)); it is not an
   area-weighted regridding, and the index says so. The pooled grid is NOT
   family 7's point-registered g025 grid (whose centres are the multiples of
   0.25°); it is offset by half a cell, because only a cell-registered grid
   has edges native pixels can be summed between.
7. **Pooled sums and counts are the native layer block-summed, exactly.** The
   pooled sum is float32(Σ over the block of the PUBLISHED native float32
   sums, in float64): every native float32 sum of float16 values is a
   multiple of 2⁻²⁴ below 2²¹ (the largest channel is MOANA's 6 × 10⁴ × 31),
   and up to 169 of them sum below 2²⁹, which float64's 53 bits hold exactly —
   so the float64 block sum is exact in any order and anyone summing the
   native files gets the same float32 bit for bit. **Pooled counts are
   uint16**: a cell pools up to 6 × 6 × 31 = 1,116 (4 km), 13 × 13 × 31 =
   5,239 (2 km) or 7 × 7 × 248 = 12,152 (IR) values — past uint8, inside
   65,535 (asserted).
8. **Pooled m2 by Chan's formula over the block**, exactly as the brief
   writes it: m2_block = Σ m2_k + Σ n_k (mean_k − mean_block)², in float64
   from the published native values, mean_block from the exact block sum;
   cast once to float32.
9. **A sibling index, not a block inside E-088's.** `src/f1data.js` matches
   every E-088 store to a registry store by `source_store` and would push a
   "the sums' grid is not the store's" error line for a 0.25° pooled grid laid
   over a 4 km store — visible on the live site. So the fine stores get
   `data/gridded_monthly_fine_index.json` (`schema: "gridded-monthly-fine/1"`)
   and `data/gridded_monthly_index.json` is not touched; nothing the live
   site reads changes until the website agent wires the new file.
10. **Box discipline as E-088 §2.8–2.9**: one store at a time; per store
    export → falsifier → Hub commit → every file streamed back and its sha256
    compared (`ml/hf_mirror.py`'s rule) → free the disk; a store whose
    `monthly/stats.json` on the Hub names the same source digest is skipped.
    Workers take runs of consecutive months; a month's bins are downloaded
    (each sha256-checked against the store's own `store.json`), decoded tile by
    tile and freed as soon as no later month of the run needs them, so disk
    in flight is about one month of bins per worker.

## 3. Layout

Per store, under `tensors/family1_gf/<store>/monthly/` on
`chfrank/earth-tensors`:

```
native/tile_grid.json                 the layout's spec (format family1-monthly-tiles/1)
native/<yyyy>/m_<yyyy>-<mm>.zst       one per (year, month) with ≥ 1 frame
native/<yyyy>/m_<yyyy>-<mm>.idx.npy   int64 [n_tiles_y, n_tiles_x, 3, 2]
pooled025/sum.npy                     [12, C, Y, Hp, Wp] float32
pooled025/count.npy                   [12, C, Y, Hp, Wp] uint16
pooled025/m2.npy                      [12, C, Y, Hp, Wp] float32
stats.json  verify.json  manifest.json
```

**Native month file.** Tiles row-major (ty, then tx); within a tile the
parts **sum** (float32), **count** (uint8), **m2** (float32), each
T × T × C in C order [row, col, channel], little-endian, one zstd frame per
(tile, part), padded at the grid's edge with zeros (count 0). The index
entry (ty, tx, part) is the 16 bytes (offset, length) at
`index_header_bytes + 16·((ty·n_tiles_x + tx)·3 + part)` (header 128 bytes
for all four stores); length 0 for all three parts = the tile holds no
observation that month (sum 0, count 0, m2 0, nothing stored). A month with
no frame has no file (`frames_present` says 0).

**Pooled files.** E-088's layout exactly: axes [month, channel, year, lat,
lon], lat south-first, 128-byte .npy header, plane (m, c, y) at
`header_len + ((m·C + c)·n_years + (y − year_first))·plane_bytes`.

**The index** `data/gridded_monthly_fine_index.json` (written by the hosted
restore, `ml/publish_fine_monthly_index.py`, never by hand). Top level:
`schema`, `restore_verified`, `cors_measured` {`sum`, `count`,
`native_shard`}, `month_rule`, `combine`, `pool_rule`, `native_layout`,
`pooled_layout`, `offset_formula`, `std_rel_tol`, `mean_bound`, `stores`.
Per store `stores["family1_gf/<name>"]`: `family`, `name`, `title`,
`licence`, `chans`, `channels` (PHYSICAL units, min/max), `units`,
`value_offset`, `source_channels`, `frame_seconds`, `years`, `year_first`,
`year_last`, `n_years`, `frames_present` / `frames_possible` ([Y][12]),
`max_count`, `record_first`, `record_last`, `source_store`,
`source_sha256_block_digest`, `falsifier`, and two layer blocks:

- `native`: `base` (URL prefix), `tile_grid` {url, bytes, sha256}, `H`, `W`,
  `C`, `tile`, `n_tiles_y`, `n_tiles_x`, `parts` [{name, dtype, itemsize,
  part_bytes}], `index_shape`, `index_dtype`, `index_header_bytes`,
  `shard_path`, `index_path`, `months_columns` = [year, month, frames,
  tiles_stored, shard_bytes, shard_sha256, index_bytes, index_sha256],
  `months` (one row per file pair), `bytes_total`. Tile extents and the native
  grid's affine rule are in `native/tile_grid.json` (copied from the store's
  own `tile_grid.json`).
- `pooled`: `grid` {H, W, deg, lat0, lon0, dlat, dlon, south_first: true,
  global_row0, global_col0, registration, rule, block_note, exact_blocks,
  rows [[r0, r1)…] per pooled row, cols [[c0, c1)…] per pooled column,
  native_north_first}, `count_note`, and `sum` / `count` / `m2` each with
  E-088's keys (`url`, `bytes`, `sha256`, `header_len`, `shape`, `dtype`,
  `itemsize`, `axes`, `plane_bytes`, `month_channel_bytes`).

Fixture: `data/gridded_monthly/fixture_fine/` — two tiny stores (an
oc4k-shaped daily north-first grid with a whole empty tile; an irtb-shaped
uint8 three-hourly south-first band with a full 248-frame month and
centre-binned cells) built with the real tier-G framework and exported,
falsified and indexed with the real code at a 15° pooled grid
(`gridded_monthly_fine_index.json`, < 0.4 MB), plus `expected.json` —
numpy's answers over the native frames (per (year, month) and over all
months, at sample native pixels and pooled cells) for the reader's tests.

## 4. Sizes

E-088 §8's probe, extended (native = measured sum + count of one month × the
months in the record, before m2; pooled = dense, computed exactly):

| store | months | native sum + count (§8) | native with m2 (estimate, ≈ 2×) | pooled 0.25° sum + count + m2 |
|---|---|---|---|---|
| `oc4k` | 304 | 37.3 GB | ≈ 70 GB | 9.70 GB (3 ch × 26 y) |
| `pace4k` | 29 | 6.1 GB | ≈ 12 GB | 2.61 GB (7 ch × 3 y) |
| `sst_acspo02` | 320 | 77.3 GB | ≈ 150 GB | 6.72 GB (2 ch × 27 y) |
| `irtb` | 345 | 9.6 GB | ≈ 19 GB | 1.20 GB (1 ch × 29 y, the band) |
| total | | 130 GB | ≈ 250 GB | 20.2 GB |

The pull is ≈ 1.17 TB of shards (`irtb` 518 GB, `sst_acspo02` 453, `oc4k`
176, `pace4k` 21). §9 replaces the estimates with measurements.

## 5. The falsifier

**On the box, before each store's upload — a failure stops that store**
(`export_fine_monthly.py verify`):

1. **Sampled whole-globe planes**: the first and last month with data and one
   month per decade the record touches, EVERY channel, EVERY native pixel and
   EVERY pooled cell: the native layer's mean / count / std and the pooled
   layer's, against numpy over the month's native frames read tile by tile
   through `sharded.ShardedGroup` from the kept, sha256-checked bins (the
   pooled reference computed directly from those frames, cells assigned from
   the grid). Counts exactly; means within the rigorous bound (E-088 §2.4;
   for the pooled layer plus Σ ½ulp of the native sums pooled into the
   cell); std within 10⁻⁶ (|mean| + std).
2. **Two period boxes** composed from their monthly planes in BOTH layers
   against numpy over every native frame of the period in the box: a whole
   mid-record year (ocean stores: 38–42° N, 42–38° W; `irtb`: 0–4° N,
   30–26° W) and a June–August season of another year (ocean: 36–32° S,
   14–18° E; `irtb`: 10–6° S, 100–104° E).
3. **The pooled layer against the native layer block-summed, EVERY month**,
   by an independent route (whole-array reduceat over the published native
   files): sums bit for bit, counts exactly, the std of m2 within the
   tolerance against m2 by the algebraic form Σm2 + Σs²/n − S²/N; and sum =
   m2 = 0 wherever the native count is 0.
4. Inside the export: every month file is read back through the reader and
   compared bit for bit with the arrays that were written.

**In the suite** (`tests/test_export_fine_monthly.py`, 20 checks): native
sums equal float32(numpy's float64 sum) of the fixture frames BIT FOR BIT,
counts exactly, m2 within the tolerance — every month, pixel and channel;
the pooled layer equals the native layer block-summed exactly by an
independent cell loop; the pooled m2 / mean / count equal numpy computed
directly at the pooled resolution from the raw frames; empty tiles absent;
the IR shape (kelvin, 248 in a uint8 count, pooled counts past 255); the
pool map of the four real grids; `verify` passes and FAILS on a nudged
native sum, pooled sum and pooled m2; a tampered shard refused; the
publisher's commits, read-back refusal, hosted index and failed-report
refusal; `run` end to end with the skip; the committed fixture; E-088's live
index untouched.

## 6. Cost

One rented Vast box (`BOX_PROFILE=assembly`, verified host, ≥ 16 CPUs —
decoding ≈ 20 core-hours of tiles is the compute — ≥ 96 GB RAM and 400 GB of
disk so the 2 km store's ≈ 150 GB native layer, its kept bins and the
workers' in-flight months fit with headroom). E-088 measured ≈ 110 MB/s from
the Hub: ≈ 3 h for 1.17 TB, plus uploading ≈ 270 GB and streaming it back.
Estimate 5–7 h at $0.2–0.6/h: **≈ $1–4**. Then the hosted restore (≈ 270 GB
streamed back on a GitHub runner) and the CORS check.

## 7. What exists

- `ml/family1/monthly_tiles.py` — the native layout: spec, writer, reader.
- `ml/export_fine_monthly.py` — `export` / `verify` / `run` / `fixture`
  (`--only YYYY-MM` for a trial; a trial is never published).
- `ml/publish_fine_monthly_index.py` — `upload_store` (box: yearly commits,
  the pooled commit, the read-back) and `index` (hosted restore, or `--local`).
- `.github/workflows/fine-monthly.yml` — `workflow_dispatch` only; job
  `export` on the rented box, job `restore` on `ubuntu-latest`; HF_TOKEN as an
  env var exactly as `gridded-monthly.yml`.
- `tests/make_fine_monthly_fixture.py`, `tests/test_export_fine_monthly.py`,
  `data/gridded_monthly/fixture_fine/`.

## 8. Status

- 2026-10-08: plan, layout module, exporter, publisher, workflow, 20 toy
  checks and the fixture written and green.
